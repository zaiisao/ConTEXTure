from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, ControlNetModel
from huggingface_hub import hf_hub_download
from transformers import CLIPTextModel, CLIPTokenizer, logging

# suppress partial model loading warning
from src import utils
from src.utils import seed_everything

logging.set_verbosity_error()
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from tqdm.auto import tqdm
import cv2
import numpy as np
from PIL import Image
import math
from einops import rearrange

import torchvision
import os

from omegaconf import OmegaConf
from .zero123.zero123.ldm.util import instantiate_from_config

class StableDiffusion(nn.Module):
    def __init__(self, device, model_name='CompVis/stable-diffusion-v1-4', concept_name=None, concept_path=None,
                 latent_mode=True,  min_timestep=0.02, max_timestep=0.98, no_noise=False,
                 use_inpaint=False, second_model_type=None, guess_mode=False):

        assert second_model_type in [None, 'zero123', 'control_zero123', 'zero123plus']

        super().__init__()

        try:
            with open('./TOKEN', 'r') as f:
                self.token = f.read().replace('\n', '')  # remove the last \n!
                logger.info(f'loaded hugging face access token from ./TOKEN!')
        except FileNotFoundError as e:
            self.token = True
            logger.warning(
                f'try to load hugging face access token from the default place, make sure you have run `huggingface-cli login`.')

        self.device = device
        self.latent_mode = latent_mode
        self.no_noise = no_noise
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * min_timestep)
        self.max_step = int(self.num_train_timesteps * max_timestep)
        self.use_inpaint = use_inpaint
        self.second_model_type = second_model_type
        self.guess_mode = guess_mode

        logger.info(f'loading stable diffusion with {model_name}...')

        # 1. Load the autoencoder model which will be used to decode the latents into image space. 
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", use_auth_token=self.token).to(self.device)

        # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder='tokenizer', use_auth_token=self.token)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder='text_encoder',
                                                          use_auth_token=self.token).to(self.device)
        self.image_encoder = None
        self.image_processor = None

        # 3. The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet", use_auth_token=self.token).to(
            self.device)

        if self.use_inpaint:
            self.inpaint_unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-inpainting",
                                                                     subfolder="unet", use_auth_token=self.token).to(
                self.device)

        if self.second_model_type == "control_zero123":
            self.second_model = self.init_zero123(control=True)
        elif self.second_model_type == "zero123":
            self.second_model = self.init_zero123(control=False)
        elif self.second_model_type == "zero123plus":
            self.zero123plus = DiffusionPipeline.from_pretrained(
                "sudo-ai/zero123plus-v1.1", custom_pipeline="sudo-ai/zero123plus-pipeline",
                torch_dtype=torch.float16
            )
            self.zero123plus.add_controlnet(ControlNetModel.from_pretrained(
                "sudo-ai/controlnet-zp11-depth-v1", torch_dtype=torch.float16
            ), conditioning_scale=2)
            # Feel free to tune the scheduler
            # pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            #     pipeline.scheduler.config, timestep_spacing='trailing'
            # )
            self.zero123plus.to(self.device)
        elif self.second_model_type is not None:
            raise NotImplementedError

        # 4. Create a scheduler for inference
        self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                       num_train_timesteps=self.num_train_timesteps, steps_offset=1,
                                       skip_prk_steps=True)
        # NOTE: Recently changed skip_prk_steps, need to see that works
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        if concept_name is not None:
            self.load_concept(concept_name, concept_path)
        logger.info(f'\t successfully loaded stable diffusion!')

    def init_zero123(self, control=True):
        if control:
            config = OmegaConf.load("./src/zero123/ControlNet/models/cldm_zero123.yaml")
        else:
            config = OmegaConf.load("./src/zero123/zero123/configs/sd-objaverse-finetune-c_concat-256.yaml")

        # pl_sd = torch.load("./src/zero123/control_zero123.ckpt", map_location='cpu')
        pl_sd = torch.load("./epoch=19-step=6359.ckpt", map_location='cpu')
        sd = pl_sd['state_dict']

        model = instantiate_from_config(config.model).to(self.device)
        model.load_state_dict(sd, strict=False)
        model.eval()

        return model

    @torch.no_grad()
    def get_zero123_inputs(
        self,
        cond_image,
        depth_map,
        x, y, z=0.7,
        n_samples=1,
        scale=1.0,
        use_control=True
    ):
        # JA: x = theta (relative elevation), y = phi (relative azimuth), z = radius
        # The reference view is the front view (phi, theta) = (0, 60)

        # JA: The following code is from gradio_new_depth_texture.py

        c = self.second_model.get_learned_conditioning(cond_image).tile(n_samples, 1, 1)

        # T = torch.tensor([math.radians(x), math.sin(
        #     math.radians(y)), math.cos(math.radians(y)), z]) # JA: In the TEXTure code, x and y are in radians already

        # Added by JA:
        # Zero123 was trained with the azimuth angle [-pi, pi], but TEXTure uses [0, 2pi].
        if y > math.pi:
            y -= 2 * math.pi

        T = torch.tensor([x, math.sin(y), math.cos(y), z])

        T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
        c = torch.cat([c, T], dim=-1)
        c = self.second_model.cc_projection(c)
        cond = {}
        cond['c_crossattn'] = [c]
        cond['c_concat'] = [self.second_model.encode_first_stage((cond_image.to(c.device))).mode().detach().repeat(n_samples, 1, 1, 1)]
        
        if use_control:
            depth_min = torch.amin(depth_map, dim=[0, 1, 2], keepdim=True)
            depth_max = torch.amax(depth_map, dim=[0, 1, 2], keepdim=True)

            control = 2. * (depth_map - depth_min) / (depth_max - depth_min) - 1.
            control = torch.stack([control for _ in range(n_samples)], dim=0)
            # control = rearrange(control, 'b h w c -> b c h w').clone()
            # control = control.repeat_interleave(3, dim=1) # (b, 1, 512, 512) -> (b, 3, 512, 512)
            control = torch.concat([control, control, control], dim=1).to(control.device)

            cond['c_control'] = [control]

        if scale != 1.0:
            h, w = depth_map.shape[-2], depth_map.shape[-1]

            uc = {}
            uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
            uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]

            if use_control:
                # uc['c_control'] = None if self.guess_mode else [control]
                uc['c_control'] = [torch.zeros_like(control).to(control.device)]
        else:
            uc = None

        return cond, uc

    def load_concept(self, concept_name, concept_path=None):
        # NOTE: No need for both name and path, they are the same!
        if concept_path is None:
            repo_id_embeds = f"sd-concepts-library/{concept_name}"
            learned_embeds_path = hf_hub_download(repo_id=repo_id_embeds, filename="learned_embeds.bin")
            # token_path = hf_hub_download(repo_id=repo_id_embeds, filename="token_identifier.txt")
            # with open(token_path, 'r') as file:
            #     placeholder_token_string = file.read()
        else:
            learned_embeds_path = concept_path

        loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")

        # separate token and the embeds
        for trained_token in loaded_learned_embeds:
            # trained_token = list(loaded_learned_embeds.keys())[0]
            print(f'Loading token for {trained_token}')
            embeds = loaded_learned_embeds[trained_token]

            # cast to dtype of text_encoder
            dtype = self.text_encoder.get_input_embeddings().weight.dtype
            embeds.to(dtype)

            # add the token in tokenizer
            token = trained_token
            num_added_tokens = self.tokenizer.add_tokens(token)
            if num_added_tokens == 0:
                raise ValueError(
                    f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer.")

            # resize the token embeddings
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))

            # get the id for the token and assign the embeds
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            self.text_encoder.get_input_embeddings().weight.data[token_id] = embeds

    def get_text_embeds(self, prompt, negative_prompt=None):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        logger.info(prompt)
        logger.info(text_input.input_ids)

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        if negative_prompt is None:
            negative_prompt = [''] * len(prompt)
        uncond_input = self.tokenizer(negative_prompt, padding='max_length',
                                      max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def img2img_single_step(self, text_embeddings, prev_latents, depth_mask, step, guidance_scale=100):
        # input is 1 3 512 512
        # depth_mask is 1 1 512 512
        # text_embeddings is 2 512

        def sample(prev_latents, depth_mask, step):
            latent_model_input = torch.cat([prev_latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input,
                                                                  step)  # NOTE: This does nothing

            latent_model_input_depth = torch.cat([latent_model_input, depth_mask], dim=1)
            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input_depth, step, encoder_hidden_states=text_embeddings)[
                    'sample']

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, step, prev_latents)['prev_sample']

            return latents

        depth_mask = F.interpolate(depth_mask, size=(64, 64), mode='bicubic',
                                   align_corners=False)

        depth_mask = 2.0 * (depth_mask - depth_mask.min()) / (depth_mask.max() - depth_mask.min()) - 1.0

        with torch.no_grad():
            target_latents = sample(prev_latents, depth_mask, step=step)
        return target_latents

    # JA: I renamed  "depth_mask" to "original_depth_mask"
    # The reason for this is because the existing code reduces the size of the depth mask, overwriting
    # the original depth map. However, the Control0123 model has been trained on the full-size depth mask.
    # inputs = Q_t = cropped_rgb_render
    def img2img_step(self, text_embeddings, inputs, original_depth_mask, guidance_scale=100, strength=0.5,
                     num_inference_steps=50, update_mask=None, latent_mode=False, check_mask=None,
                     fixed_seed=None, check_mask_iters=0.5, intermediate_vis=False, view_dir=None,
                     front_image=None, phi=None, theta=None, condition_guidance_scales=None):
        # input is 1 3 512 512      # JA: inputs is cropped_rgb_render.detach()
        # depth_mask is 1 1 512 512
        # text_embeddings is 2 512 # JA: text_embeddings contains the single embedding for one of the six view prompts

        # JA: We need to replace text_embeddings by the embedding vector of the cond image + the relative camera pose
        # We also set the self.unet 

        intermediate_results = []

        def sample(latents, depth_mask, strength, num_inference_steps, update_mask=None, check_mask=None,
                   masked_latents=None):
            self.scheduler.set_timesteps(num_inference_steps)
            noise = None
            if latents is None:
                # Last chanel is reserved for depth
                latents = torch.randn(
                    ( # JA: text_embeddings is a global variable of the sample inner function
                        text_embeddings.shape[0] // 2, self.unet.in_channels - 1, depth_mask.shape[2],
                        depth_mask.shape[3]),
                    device=self.device)
                timesteps = self.scheduler.timesteps
            else: # JA: latents is the latent version of Q_t without noise
                # Strength has meaning only when latents are given
                timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength)
                latent_timestep = timesteps[:1] # JA: The first timestep; in our case, tensor([981])
                if fixed_seed is not None:
                    seed_everything(fixed_seed)
                noise = torch.randn_like(latents)
                if update_mask is not None:
                    # NOTE: I think we might want to use same noise?
                    gt_latents = latents # JA: ground truth will be used to calculate the noised truth later. gt_latents refers to Q_t in latent space without noise and has a shape of (64, 64)
                    latents = torch.randn(
                        (text_embeddings.shape[0] // 2, self.unet.in_channels - 1, depth_mask.shape[2],
                         depth_mask.shape[3]),
                        device=self.device)
                else:
                    latents = self.scheduler.add_noise(latents, noise, latent_timestep)
            # JA: In our experiment, the latents is a random tensor at this point

            depth_mask = torch.cat([depth_mask] * 2) # JA: depth_mask is D_t (in latent space 64x64)

            # print(f"zero123, azimuth: {phi}, overhead: {theta}")

            with torch.autocast('cuda'):
                for i, t in tqdm(enumerate(timesteps)): # JA: denoising iteration loop of the sample function
                    is_inpaint_range = self.use_inpaint and (10 < i < 20)
                    mask_constraints_iters = True  # i < 20
                    is_inpaint_iter = is_inpaint_range  # and i %2 == 1

                    if is_inpaint_iter:
                        # JA: inpaint pipeline

                        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                        latent_model_input = torch.cat([latents] * 2)
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input,
                                                                            t)  # NOTE: This does nothing

                        latent_mask = torch.cat([update_mask] * 2)
                        latent_image = torch.cat([masked_latents] * 2) # JA: latent_image is the masked latent image
                        latent_model_input_inpaint = torch.cat([latent_model_input, latent_mask, latent_image], dim=1)
                        with torch.no_grad():
                            noise_pred_inpaint = \
                                self.inpaint_unet(latent_model_input_inpaint, t, encoder_hidden_states=text_embeddings)[
                                    'sample']   # JA: When we use Zero123, the text embeddings should be replaced with the
                                                # embeddings of the cond image plus the relative camera pose
                            noise_pred = noise_pred_inpaint

                            # JA: Although there are six text embeddings, we generate 10 images because we have
                            # 10 different inpaint masks.

                        # perform guidance
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    else:
                        # JA: depth pipeline

                        if mask_constraints_iters and update_mask is not None:
                            noised_truth = self.scheduler.add_noise(gt_latents, noise, t) # JA: noised_truth is z_Q_t and gt_latents is z_Q_0 (00XX_cropped_input.jpg)
                            # JA: update_mask and check_mask are used in both the inpainting and depth pipelines
                            # This implements formula 2 of the paper.
                            if check_mask is not None and i < int(len(timesteps) * check_mask_iters):
                                curr_mask = check_mask
                            else:
                                curr_mask = update_mask # JA: update_mask means "refine" in the paper

                            # JA: This corresponds to the formula 1 of the equation paper.
                            # z_i ← z_i * m_blended + z_Q_t * (1 − m_blended)
                            # m_blended is curr_mask, which indicates the fill-in/inpaint location
                            # (1 - curr_mask) is the background
                            # On the right side, the latents is the random tensor and the noised_truth is the ground
                            # truth with some noise. latents now refers to the image being denoised and plays the role of
                            # x in apply_model.

                            # torchvision.utils.save_image(self.decode_latents(latents), f"./texture_test/{round(math.degrees(phi))}_{round(math.degrees(theta))}_{i}_latents_before.png")
                            # torchvision.utils.save_image(self.decode_latents(noised_truth), f"./texture_test/{round(math.degrees(phi))}_{round(math.degrees(theta))}_{i}_noised_truth.png")
                            # torchvision.utils.save_image(F.interpolate(curr_mask, size=(image_size, image_size))[0], f"./texture_test/{round(math.degrees(phi))}_{round(math.degrees(theta))}_{i}_curr_mask.png")
                            # # torchvision.utils.save_image(F.interpolate(original_depth_mask, size=(512, 512))[0], f"./texture_test/{round(math.degrees(phi))}_{round(math.degrees(theta))}_{i}_depth_mask.png")
                            # torchvision.utils.save_image(pred_rgb_small, f"./texture_test/{round(math.degrees(phi))}_{round(math.degrees(theta))}_{i}_pred_rgb_512.png")

                            # # JA: This blend operation is executed for the traditional depth pipeline and the zero123 pipeline
                            # latents = latents * curr_mask + noised_truth * (1 - curr_mask)
                            # torchvision.utils.save_image(self.decode_latents(latents), f"./texture_test/{round(math.degrees(phi))}_{round(math.degrees(theta))}_{i}_latents_after.png")

                            # debug_image_paths = [
                            #     f"./texture_test/{round(math.degrees(phi))}_{round(math.degrees(theta))}_{i}_latents_before.png",
                            #     f"./texture_test/{round(math.degrees(phi))}_{round(math.degrees(theta))}_{i}_pred_rgb_512.png",
                            #     f"./texture_test/{round(math.degrees(phi))}_{round(math.degrees(theta))}_{i}_noised_truth.png",
                            #     f"./texture_test/{round(math.degrees(phi))}_{round(math.degrees(theta))}_{i}_curr_mask.png",
                            #     f"./texture_test/{round(math.degrees(phi))}_{round(math.degrees(theta))}_{i}_latents_after.png"
                            # ]
                            # images = [Image.open(x) for x in debug_image_paths]
                            # widths, heights = zip(*(i.size for i in images))

                            # total_width = sum(widths)
                            # max_height = max(heights)

                            # new_im = Image.new('RGB', (total_width, max_height))

                            # x_offset = 0
                            # for im in images:
                            #     new_im.paste(im, (x_offset,0))
                            #     x_offset += im.size[0]

                            # new_im.save(f'./texture_test/{round(math.degrees(phi))}_{round(math.degrees(theta))}_{i}.png')

                            # for image_path in debug_image_paths:
                            #     os.remove(image_path)
                        # JA: latents is random initially

                        if self.second_model_type is None or view_dir == "front":
                            # JA: SD 2.0 depth pipeline

                            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                            latent_model_input = torch.cat([latents] * 2)
                            latent_model_input = self.scheduler.scale_model_input(latent_model_input,
                                                                                t)  # NOTE: This does nothing

                            latent_model_input_depth = torch.cat([latent_model_input, depth_mask], dim=1)
                            # predict the noise residual
                            with torch.no_grad():
                                noise_pred = self.unet(latent_model_input_depth, t, encoder_hidden_states=text_embeddings)[
                                    'sample']
                                        # JA: Although there are six text embeddings, we generate 10 images because we have
                                        # 10 depth maps.

                            # perform guidance
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                        else:
                            # JA: zero123 or control zero123

                            # JA: The following block was added to handle the control image for zero123
                            use_control = self.second_model_type == "control_zero123"

                            cond, uc = self.get_zero123_inputs(
                                F.interpolate(front_image, size=(image_size, image_size)),
                                F.interpolate(original_depth_mask, size=(image_size, image_size))[0],
                                theta, phi,
                                scale=guidance_scale,
                                use_control=use_control
                            )

                            t = t[None].to(self.device)
                            with torch.no_grad():
                                # JA: Note that latents -- not latent_model_input -- goes into each
                                # apply_model call, because latent_model_input is created on the
                                # assumption that cond and uncond will be sampled at the same time

                                if uc is None: # JA: We do not consider the negative direction of the unconditional/random generation
                                    # model_t = model_uncond = self.second_model.apply_model(latents, t, cond)
                                    noise_pred = self.second_model.apply_model(latents, t, cond)
                                else:
                                    # JA: We separate conditional generation and unconditional generation because the concatenating uncond
                                    # and cond raises a type mismatch error because cond can contain None value for c_control and None is
                                    # not a tensor


                                    # if not individual_control_of_conditions:
                                    if condition_guidance_scales is None:
                                        zero123_guidance_scale = 3#condition_guidance_scales["all"]
                                        model_uncond_all = self.second_model.apply_model(latents, t, uc)
                                        model_t = self.second_model.apply_model(latents, t, cond)
                                        noise_pred = model_uncond_all + zero123_guidance_scale * (model_t - model_uncond_all)
                                    else:
                                        model_concat_control_no_crossattn = self.second_model.apply_model(latents, t, {
                                            "c_crossattn": uc["c_crossattn"],
                                            "c_concat": cond["c_concat"],
                                            "c_control": cond["c_control"]
                                        })

                                        model_concat_no_control_no_crossattn = self.second_model.apply_model(latents, t, {
                                            "c_crossattn": uc["c_crossattn"],
                                            "c_concat": cond["c_concat"],
                                            "c_control": uc["c_control"]
                                        })

                                        model_concat_control_crossattn = self.second_model.apply_model(latents, t, {
                                            "c_crossattn": cond["c_crossattn"],
                                            "c_concat": cond["c_concat"],
                                            "c_control": cond["c_control"]
                                        })

                                        guidance_i = condition_guidance_scales["i"]
                                        guidance_t = condition_guidance_scales["t"]

                                        noise_pred = model_concat_no_control_no_crossattn \
                                            + guidance_i * (model_concat_control_no_crossattn - model_concat_no_control_no_crossattn) \
                                            + guidance_t * (model_concat_control_crossattn - model_concat_control_no_crossattn)

                                        # noise_pred = model_uncond + guidance_scale_all * (model_t - model_uncond)
                                        #            = a + CFG * (b - a)

                                        # model_uncond = the unconditional prediction with all three conditions set to None
                                        # model_uncond_concat = the unconditional prediction with only concat condition set to None

                    # compute the previous noisy sample x_t -> x_t-1

                    if intermediate_vis:
                        vis_alpha_t = torch.sqrt(self.scheduler.alphas_cumprod)
                        vis_sigma_t = torch.sqrt(1 - self.scheduler.alphas_cumprod)
                        a_t, s_t = vis_alpha_t[t], vis_sigma_t[t]
                        vis_latents = (latents - s_t * noise) / a_t
                        vis_latents = 1 / 0.18215 * vis_latents
                        image = self.vae.decode(vis_latents).sample
                        image = (image / 2 + 0.5).clamp(0, 1)
                        image = image.cpu().permute(0, 2, 3, 1).numpy()
                        image = Image.fromarray((image[0] * 255).round().astype("uint8"))
                        intermediate_results.append(image)

                    # JA: Denoise one step. This is applied for every pipeline at each iteration
                    latents = self.scheduler.step(noise_pred, t, latents)['prev_sample'] # return x_prev, pred_x0

            return latents
        # JA: end of sample function

        if True: #self.second_model_type is None or view_dir == "front":
            image_size = 512
        else:
            image_size = 256 # JA: We use image size of 256 when producing the image from non-front views using zero123 or control zero123

        depth_mask = F.interpolate(original_depth_mask, size=(image_size // 8, image_size // 8), mode='bicubic',
                                   align_corners=False) # JA: original_depth_mask is D_t (in pixel space) and has a shape of (827, 827) and is a nonzero region of the depth image
        masked_latents = None
        if inputs is None:
            latents = None
        elif latent_mode:
            latents = inputs
        else:
            # JA: inputs is the "cropped_input" which represents the non-zero region of the rendered image of the current texture atlas
            pred_rgb_small = F.interpolate(inputs, (image_size, image_size), mode='bilinear',
                                         align_corners=False) # JA: Shape of inputs is (827, 827)

            # if self.second_model_type in ["zero123", "control_zero123"] and view_dir != "front":
            #     latents = None#torch.randn((64, 64), device=self.device)
            # else:
            latents = self.encode_imgs(pred_rgb_small) # JA: Convert the rgb_render_output to the latent space of shape 64x64
            # def encode_imgs(self, imgs):
            #       # imgs: [B, 3, H, W]

            #          imgs = 2 * imgs - 1

            #          posterior = self.vae.encode(imgs).latent_dist
            #          latents = posterior.sample() * 0.18215

            if self.use_inpaint:
                update_mask_small = F.interpolate(update_mask, (image_size, image_size))
                masked_inputs = pred_rgb_small * (update_mask_small < 0.5) + 0.5 * (update_mask_small >= 0.5)
                masked_latents = self.encode_imgs(masked_inputs)

        if update_mask is not None:
            update_mask = F.interpolate(update_mask, (image_size // 8, image_size // 8), mode='nearest')
        if check_mask is not None:
            check_mask = F.interpolate(check_mask, (image_size // 8, image_size // 8), mode='nearest')

        # JA: Normalize depth map so that its values range from -1 to +1
        depth_mask = 2.0 * (depth_mask - depth_mask.min()) / (depth_mask.max() - depth_mask.min()) - 1.0

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        # t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
        t = (self.min_step + self.max_step) // 2

        with torch.no_grad():
            # JA: target_latents is the denoised image in the latent space for the given text_embeddings
            target_latents = sample(latents, depth_mask, strength=strength, num_inference_steps=num_inference_steps,
                                    update_mask=update_mask, check_mask=check_mask, masked_latents=masked_latents)
            target_rgb = self.decode_latents(target_latents)    # JA: Convert into the pixel space. target_rgb is the image corresponding to a specific view prompt
                                                                # In our case, we need to obtain the image corresponding to a specific relative camera pose which
                                                                # is obtained from the front view image. In our case target_rgb is the image created from zero123
                                                                # by means of the relative camera pose.

        # if image_size == 256:
        #     target_rgb = F.interpolate(target_rgb, (512, 512))

        if latent_mode:
            return target_rgb, target_latents # JA: The target_rgb is the result from denoising the blended latent
        else:
            return target_rgb, intermediate_results
        
    def zero123plus_img2img_step(self, text_embeddings, inputs, original_depth_mask, guidance_scale=100, strength=0.5,
                     num_inference_steps=50, update_mask=None, latent_mode=False, check_mask=None,
                     fixed_seed=None, check_mask_iters=0.5, intermediate_vis=False, view_dir=None,
                     front_image=None, phi=None, theta=None, condition_guidance_scales=None):
        # input is 1 3 512 512      # JA: inputs is cropped_rgb_render.detach()
        # depth_mask is 1 1 512 512
        # text_embeddings is 2 512 # JA: text_embeddings contains the single embedding for one of the six view prompts

        intermediate_results = []

        def sample(latents, depth_mask, strength, num_inference_steps, update_mask=None, check_mask=None,
                   masked_latents=None):
            self.scheduler.set_timesteps(num_inference_steps)
            noise = None
            if latents is None:
                # Last chanel is reserved for depth
                latents = torch.randn(
                    ( # JA: text_embeddings is a global variable of the sample inner function
                        text_embeddings.shape[0] // 2, self.unet.in_channels - 1, depth_mask.shape[2],
                        depth_mask.shape[3]),
                    device=self.device)
                timesteps = self.scheduler.timesteps
            else: # JA: latents is the latent version of Q_t without noise
                # Strength has meaning only when latents are given
                timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength)
                latent_timestep = timesteps[:1] # JA: The first timestep; in our case, tensor([981])
                if fixed_seed is not None:
                    seed_everything(fixed_seed)
                noise = torch.randn_like(latents)
                if update_mask is not None:
                    # NOTE: I think we might want to use same noise?
                    gt_latents = latents # JA: ground truth will be used to calculate the noised truth later. gt_latents refers to Q_t in latent space without noise and has a shape of (64, 64)
                    latents = torch.randn(
                        (text_embeddings.shape[0] // 2, self.unet.in_channels - 1, depth_mask.shape[2],
                         depth_mask.shape[3]),
                        device=self.device)
                else:
                    latents = self.scheduler.add_noise(latents, noise, latent_timestep)
            # JA: In our experiment, the latents is a random tensor at this point

            depth_mask = torch.cat([depth_mask] * 2) # JA: depth_mask is D_t (in latent space 64x64)

            # print(f"zero123, azimuth: {phi}, overhead: {theta}")

            with torch.autocast('cuda'):
                for i, t in tqdm(enumerate(timesteps)): # JA: denoising iteration loop of the sample function
                    is_inpaint_range = self.use_inpaint and (10 < i < 20)
                    mask_constraints_iters = True  # i < 20
                    is_inpaint_iter = is_inpaint_range  # and i %2 == 1

                    if is_inpaint_iter:
                        # JA: inpaint pipeline

                        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                        latent_model_input = torch.cat([latents] * 2)
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input,
                                                                            t)  # NOTE: This does nothing

                        latent_mask = torch.cat([update_mask] * 2)
                        latent_image = torch.cat([masked_latents] * 2) # JA: latent_image is the masked latent image
                        latent_model_input_inpaint = torch.cat([latent_model_input, latent_mask, latent_image], dim=1)
                        with torch.no_grad():
                            noise_pred_inpaint = \
                                self.inpaint_unet(latent_model_input_inpaint, t, encoder_hidden_states=text_embeddings)[
                                    'sample']   # JA: When we use Zero123, the text embeddings should be replaced with the
                                                # embeddings of the cond image plus the relative camera pose
                            noise_pred = noise_pred_inpaint

                            # JA: Although there are six text embeddings, we generate 10 images because we have
                            # 10 different inpaint masks.

                        # perform guidance
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    else:
                        # JA: depth pipeline

                        if mask_constraints_iters and update_mask is not None:
                            noised_truth = self.scheduler.add_noise(gt_latents, noise, t) # JA: noised_truth is z_Q_t and gt_latents is z_Q_0 (00XX_cropped_input.jpg)
                            # JA: update_mask and check_mask are used in both the inpainting and depth pipelines
                            # This implements formula 2 of the paper.
                            if check_mask is not None and i < int(len(timesteps) * check_mask_iters):
                                curr_mask = check_mask
                            else:
                                curr_mask = update_mask # JA: update_mask means "refine" in the paper

                            # JA: This corresponds to the formula 1 of the equation paper.
                            # z_i ← z_i * m_blended + z_Q_t * (1 − m_blended)
                            # m_blended is curr_mask, which indicates the fill-in/inpaint location
                            # (1 - curr_mask) is the background
                            # On the right side, the latents is the random tensor and the noised_truth is the ground
                            # truth with some noise. latents now refers to the image being denoised and plays the role of
                            # x in apply_model.

                            torchvision.utils.save_image(self.decode_latents(latents), f"./texture_test/{round(math.degrees(phi))}_{round(math.degrees(theta))}_{i}_latents_before.png")
                            torchvision.utils.save_image(self.decode_latents(noised_truth), f"./texture_test/{round(math.degrees(phi))}_{round(math.degrees(theta))}_{i}_noised_truth.png")
                            torchvision.utils.save_image(F.interpolate(curr_mask, size=(image_size, image_size))[0], f"./texture_test/{round(math.degrees(phi))}_{round(math.degrees(theta))}_{i}_curr_mask.png")
                            # torchvision.utils.save_image(F.interpolate(original_depth_mask, size=(512, 512))[0], f"./texture_test/{round(math.degrees(phi))}_{round(math.degrees(theta))}_{i}_depth_mask.png")
                            torchvision.utils.save_image(pred_rgb_small, f"./texture_test/{round(math.degrees(phi))}_{round(math.degrees(theta))}_{i}_pred_rgb_512.png")

                            # JA: This blend operation is executed for the traditional depth pipeline and the zero123 pipeline
                            latents = latents * curr_mask + noised_truth * (1 - curr_mask)
                            torchvision.utils.save_image(self.decode_latents(latents), f"./texture_test/{round(math.degrees(phi))}_{round(math.degrees(theta))}_{i}_latents_after.png")

                            debug_image_paths = [
                                f"./texture_test/{round(math.degrees(phi))}_{round(math.degrees(theta))}_{i}_latents_before.png",
                                f"./texture_test/{round(math.degrees(phi))}_{round(math.degrees(theta))}_{i}_pred_rgb_512.png",
                                f"./texture_test/{round(math.degrees(phi))}_{round(math.degrees(theta))}_{i}_noised_truth.png",
                                f"./texture_test/{round(math.degrees(phi))}_{round(math.degrees(theta))}_{i}_curr_mask.png",
                                f"./texture_test/{round(math.degrees(phi))}_{round(math.degrees(theta))}_{i}_latents_after.png"
                            ]
                            images = [Image.open(x) for x in debug_image_paths]
                            widths, heights = zip(*(i.size for i in images))

                            total_width = sum(widths)
                            max_height = max(heights)

                            new_im = Image.new('RGB', (total_width, max_height))

                            x_offset = 0
                            for im in images:
                                new_im.paste(im, (x_offset,0))
                                x_offset += im.size[0]

                            new_im.save(f'./texture_test/{round(math.degrees(phi))}_{round(math.degrees(theta))}_{i}.png')

                            for image_path in debug_image_paths:
                                os.remove(image_path)
                        # JA: latents is random initially

                        if self.second_model_type is None or view_dir == "front":
                            # JA: SD 2.0 depth pipeline

                            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                            latent_model_input = torch.cat([latents] * 2)
                            latent_model_input = self.scheduler.scale_model_input(latent_model_input,
                                                                                t)  # NOTE: This does nothing

                            latent_model_input_depth = torch.cat([latent_model_input, depth_mask], dim=1)
                            # predict the noise residual
                            with torch.no_grad():
                                noise_pred = self.unet(latent_model_input_depth, t, encoder_hidden_states=text_embeddings)[
                                    'sample']
                                        # JA: Although there are six text embeddings, we generate 10 images because we have
                                        # 10 depth maps.

                            # perform guidance
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                        else:
                            # JA: zero123 or control zero123

                            # JA: The following block was added to handle the control image for zero123
                            use_control = self.second_model_type == "control_zero123"

                            cond, uc = self.get_zero123_inputs(
                                F.interpolate(front_image, size=(image_size, image_size)),
                                F.interpolate(original_depth_mask, size=(image_size, image_size))[0],
                                theta, phi,
                                scale=guidance_scale,
                                use_control=use_control
                            )

                            t = t[None].to(self.device)
                            with torch.no_grad():
                                # JA: Note that latents -- not latent_model_input -- goes into each
                                # apply_model call, because latent_model_input is created on the
                                # assumption that cond and uncond will be sampled at the same time

                                if uc is None: # JA: We do not consider the negative direction of the unconditional/random generation
                                    # model_t = model_uncond = self.second_model.apply_model(latents, t, cond)
                                    noise_pred = self.second_model.apply_model(latents, t, cond)
                                else:
                                    # JA: We separate conditional generation and unconditional generation because the concatenating uncond
                                    # and cond raises a type mismatch error because cond can contain None value for c_control and None is
                                    # not a tensor


                                    # if not individual_control_of_conditions:
                                    if condition_guidance_scales is None:
                                        zero123_guidance_scale = 3#condition_guidance_scales["all"]
                                        model_uncond_all = self.second_model.apply_model(latents, t, uc)
                                        model_t = self.second_model.apply_model(latents, t, cond)
                                        noise_pred = model_uncond_all + zero123_guidance_scale * (model_t - model_uncond_all)
                                    else:
                                        model_concat_control_no_crossattn = self.second_model.apply_model(latents, t, {
                                            "c_crossattn": uc["c_crossattn"],
                                            "c_concat": cond["c_concat"],
                                            "c_control": cond["c_control"]
                                        })

                                        model_concat_no_control_no_crossattn = self.second_model.apply_model(latents, t, {
                                            "c_crossattn": uc["c_crossattn"],
                                            "c_concat": cond["c_concat"],
                                            "c_control": uc["c_control"]
                                        })

                                        model_concat_control_crossattn = self.second_model.apply_model(latents, t, {
                                            "c_crossattn": cond["c_crossattn"],
                                            "c_concat": cond["c_concat"],
                                            "c_control": cond["c_control"]
                                        })

                                        guidance_i = condition_guidance_scales["i"]
                                        guidance_t = condition_guidance_scales["t"]

                                        noise_pred = model_concat_no_control_no_crossattn \
                                            + guidance_i * (model_concat_control_no_crossattn - model_concat_no_control_no_crossattn) \
                                            + guidance_t * (model_concat_control_crossattn - model_concat_control_no_crossattn)

                                        # noise_pred = model_uncond + guidance_scale_all * (model_t - model_uncond)
                                        #            = a + CFG * (b - a)

                                        # model_uncond = the unconditional prediction with all three conditions set to None
                                        # model_uncond_concat = the unconditional prediction with only concat condition set to None

                    # compute the previous noisy sample x_t -> x_t-1

                    if intermediate_vis:
                        vis_alpha_t = torch.sqrt(self.scheduler.alphas_cumprod)
                        vis_sigma_t = torch.sqrt(1 - self.scheduler.alphas_cumprod)
                        a_t, s_t = vis_alpha_t[t], vis_sigma_t[t]
                        vis_latents = (latents - s_t * noise) / a_t
                        vis_latents = 1 / 0.18215 * vis_latents
                        image = self.vae.decode(vis_latents).sample
                        image = (image / 2 + 0.5).clamp(0, 1)
                        image = image.cpu().permute(0, 2, 3, 1).numpy()
                        image = Image.fromarray((image[0] * 255).round().astype("uint8"))
                        intermediate_results.append(image)

                    # JA: Denoise one step. This is applied for every pipeline at each iteration
                    latents = self.scheduler.step(noise_pred, t, latents)['prev_sample'] # return x_prev, pred_x0

            return latents
        # JA: end of sample function

        if True: #self.second_model_type is None or view_dir == "front":
            image_size = 512
        else:
            image_size = 256 # JA: We use image size of 256 when producing the image from non-front views using zero123 or control zero123

        depth_mask = F.interpolate(original_depth_mask, size=(image_size // 8, image_size // 8), mode='bicubic',
                                   align_corners=False) # JA: original_depth_mask is D_t (in pixel space) and has a shape of (827, 827) and is a nonzero region of the depth image
        masked_latents = None
        if inputs is None:
            latents = None
        elif latent_mode:
            latents = inputs
        else:
            # JA: inputs is the "cropped_input" which represents the non-zero region of the rendered image of the current texture atlas
            pred_rgb_small = F.interpolate(inputs, (image_size, image_size), mode='bilinear',
                                         align_corners=False) # JA: Shape of inputs is (827, 827)

            # if self.second_model_type in ["zero123", "control_zero123"] and view_dir != "front":
            #     latents = None#torch.randn((64, 64), device=self.device)
            # else:
            latents = self.encode_imgs(pred_rgb_small) # JA: Convert the rgb_render_output to the latent space of shape 64x64

            if self.use_inpaint:
                update_mask_small = F.interpolate(update_mask, (image_size, image_size))
                masked_inputs = pred_rgb_small * (update_mask_small < 0.5) + 0.5 * (update_mask_small >= 0.5)
                masked_latents = self.encode_imgs(masked_inputs)

        if update_mask is not None:
            update_mask = F.interpolate(update_mask, (image_size // 8, image_size // 8), mode='nearest')
        if check_mask is not None:
            check_mask = F.interpolate(check_mask, (image_size // 8, image_size // 8), mode='nearest')

        # JA: Normalize depth map so that its values range from -1 to +1
        depth_mask = 2.0 * (depth_mask - depth_mask.min()) / (depth_mask.max() - depth_mask.min()) - 1.0

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        # t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
        t = (self.min_step + self.max_step) // 2

        with torch.no_grad():
            # JA: target_latents is the denoised image in the latent space for the given text_embeddings
            target_latents = sample(latents, depth_mask, strength=strength, num_inference_steps=num_inference_steps,
                                    update_mask=update_mask, check_mask=check_mask, masked_latents=masked_latents)
            target_rgb = self.decode_latents(target_latents)    # JA: Convert into the pixel space. target_rgb is the image corresponding to a specific view prompt
                                                                # In our case, we need to obtain the image corresponding to a specific relative camera pose which
                                                                # is obtained from the front view image. In our case target_rgb is the image created from zero123
                                                                # by means of the relative camera pose.

        # if image_size == 256:
        #     target_rgb = F.interpolate(target_rgb, (512, 512))

        if latent_mode:
            return target_rgb, target_latents # JA: The target_rgb is the result from denoising the blended latent
        else:
            return target_rgb, intermediate_results

    def train_step(self, text_embeddings, inputs, depth_mask, guidance_scale=100):

        # interp to 512x512 to be fed into vae.

        # _t = time.time()
        if not self.latent_mode:
            # latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False)
            pred_rgb_512 = F.interpolate(inputs, (512, 512), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_512)
            depth_mask = F.interpolate(depth_mask, size=(64, 64), mode='bicubic',
                                       align_corners=False)
        else:
            latents = inputs

        depth_mask = 2.0 * (depth_mask - depth_mask.min()) / (depth_mask.max() - depth_mask.min()) - 1.0
        # depth_mask = F.interpolate(depth_mask, size=(64,64), mode='bicubic',
        #                            align_corners=False)
        depth_mask = torch.cat([depth_mask] * 2)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # encode image into latents with vae, requires grad!
        # _t = time.time()

        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad():
            # add noise
            if self.no_noise:
                noise = torch.zeros_like(latents)
                latents_noisy = latents
            else:
                noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            # add depth
            latent_model_input = torch.cat([latent_model_input, depth_mask], dim=1)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), alpha_t * sigma_t^2
        w = (1 - self.alphas[t])
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w * (noise_pred - noise)

        # clip grad for stable training?
        # grad = grad.clamp(-1, 1)
        grad = torch.nan_to_num(grad)

        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        # _t = time.time()
        latents.backward(gradient=grad, retain_graph=True)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: backward {time.time() - _t:.4f}s')

        return 0  # dummy loss value

    def produce_latents(self, text_embeddings, depth_mask, height=512, width=512, num_inference_steps=50,
                        guidance_scale=7.5, latents=None, strength=0.5):

        self.scheduler.set_timesteps(num_inference_steps)

        if latents is None:
            # Last chanel is reserved for depth
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels - 1, height // 8, width // 8),
                                  device=self.device)
            timesteps = self.scheduler.timesteps
        else:
            # Strength has meaning only when latents are given
            timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength)
            # Dont really have to tie the scheudler to the strength
            latent_timestep = timesteps[:1]
            noise = torch.randn_like(latents)
            latents = self.scheduler.add_noise(latents, noise, latent_timestep)

        depth_mask = torch.cat([depth_mask] * 2)
        with torch.autocast('cuda'):
            for i, t in tqdm(enumerate(timesteps)):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)  # NOTE: This does nothing
                latent_model_input = torch.cat([latent_model_input, depth_mask], dim=1)
                # Depth should be added here

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents

    def decode_latents(self, latents):
        # latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False)
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def get_timesteps(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start

    def prompt_to_img(self, prompts, depth_mask, height=512, width=512, num_inference_steps=50, guidance_scale=7.5,
                      latents=None, strength=0.5):

        if isinstance(prompts, str):
            prompts = [prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts)  # [2, 77, 768]
        # new should be torch.Size([2, 77, 1024])

        # depth is in range of 20-1500 of size 1x384x384, normalized to -1 to 1, mean was -0.6
        # Resized to 64x64 # TODO: Understand range here
        depth_mask = 2.0 * (depth_mask - depth_mask.min()) / (depth_mask.max() - depth_mask.min()) - 1.0
        depth_mask = F.interpolate(depth_mask.unsqueeze(1), size=(height // 8, width // 8), mode='bicubic',
                                   align_corners=False)

        # Added as an extra channel to the latents

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, depth_mask=depth_mask, height=height, width=width, latents=latents,
                                       num_inference_steps=num_inference_steps,
                                       guidance_scale=guidance_scale, strength=strength)  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs
