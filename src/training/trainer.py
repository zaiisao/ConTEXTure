from pathlib import Path
from typing import Any, Dict, Union, List

import cv2
import einops
import imageio
import numpy as np
import pyrallis
import torch
import torch.nn.functional as F
from PIL import Image
from loguru import logger
from matplotlib import cm
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import torchvision
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, ControlNetModel

from src import utils
from src.configs.train_config import TrainConfig
from src.models.textured_mesh import TexturedMeshModel
from src.stable_diffusion_depth import StableDiffusion
from src.training.views_dataset import Zero123PlusDataset, ViewsDataset, MultiviewDataset
from src.utils import make_path, tensor2numpy, pad_tensor_to_size, split_zero123plus_grid


class TEXTure:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.paint_step = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        utils.seed_everything(self.cfg.optim.seed)

        # Make view_dirs
        self.exp_path = make_path(self.cfg.log.exp_dir)
        self.ckpt_path = make_path(self.exp_path / 'checkpoints')
        self.train_renders_path = make_path(self.exp_path / 'vis' / 'train')
        self.eval_renders_path = make_path(self.exp_path / 'vis' / 'eval')
        self.final_renders_path = make_path(self.exp_path / 'results')

        self.init_logger()
        pyrallis.dump(self.cfg, (self.exp_path / 'config.yaml').open('w'))

        self.view_dirs = ['front', 'left', 'back', 'right', 'overhead', 'bottom'] # self.view_dirs[dir] when dir = [4] = [right]
        self.mesh_model = self.init_mesh_model()
        self.diffusion = self.init_diffusion()

        if self.cfg.guide.use_zero123plus:
            self.zero123plus = self.init_zero123plus()

        self.text_z, self.text_string = self.calc_text_embeddings()
        self.dataloaders = self.init_dataloaders()
        self.back_im = torch.Tensor(np.array(Image.open(self.cfg.guide.background_img).convert('RGB'))).to(
            self.device).permute(2, 0,
                                 1) / 255.0

        self.zero123_front_input = None

        logger.info(f'Successfully initialized {self.cfg.log.exp_name}')
        
        
    #MJ:    Moved out of the loop; Currently, in every loop the def statement itself defined. The inner functions are also hard to read
    def scale_latents(self,latents): #MJ: move out of the loop
        latents = (latents - 0.22) * 0.75
        return latents

    def unscale_latents(self,latents):
        latents = latents / 0.75 + 0.22
        return latents

    def scale_image(self,image):
        image = image * 0.5 / 0.8
        return image

    def unscale_image(self,image):
        image = image / 0.5 * 0.8
        return image


    def init_mesh_model(self) -> nn.Module:
        # fovyangle = np.pi / 6 if self.cfg.guide.use_zero123plus else np.pi / 3
        fovyangle = np.pi / 3
        cache_path = Path('cache') / Path(self.cfg.guide.shape_path).stem
        cache_path.mkdir(parents=True, exist_ok=True)
        model = TexturedMeshModel(self.cfg.guide, device=self.device,
                                  render_grid_size=self.cfg.render.train_grid_size,
                                  cache_path=cache_path,
                                  texture_resolution=self.cfg.guide.texture_resolution,
                                  augmentations=False,
                                  fovyangle=fovyangle)

        model = model.to(self.device)
        logger.info(
            f'Loaded Mesh, #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
        logger.info(model)
        return model

    def init_diffusion(self) -> Any:
        # JA: The StableDiffusion class composes a pipeline by using individual components such as VAE encoder,
        # CLIP encoder, and UNet
        second_model_type = self.cfg.guide.second_model_type
        if self.cfg.guide.use_zero123plus:
            second_model_type = "zero123plus"

        diffusion_model = StableDiffusion(self.device, model_name=self.cfg.guide.diffusion_name,
                                          concept_name=self.cfg.guide.concept_name,
                                          concept_path=self.cfg.guide.concept_path,
                                          latent_mode=False,
                                          min_timestep=self.cfg.optim.min_timestep,
                                          max_timestep=self.cfg.optim.max_timestep,
                                          no_noise=self.cfg.optim.no_noise,
                                          use_inpaint=True,
                                          second_model_type=self.cfg.guide.second_model_type,
                                          guess_mode=self.cfg.guide.guess_mode)

        for p in diffusion_model.parameters():
            p.requires_grad = False
        return diffusion_model
    
    def init_zero123plus(self) -> DiffusionPipeline:
        pipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.1", custom_pipeline="sudo-ai/zero123plus-pipeline",
            torch_dtype=torch.float16
        )

        pipeline.add_controlnet(ControlNetModel.from_pretrained(
            "sudo-ai/controlnet-zp11-depth-v1", torch_dtype=torch.float16
        ), conditioning_scale=2)

        pipeline.to(self.device)

        return pipeline

    def calc_text_embeddings(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        ref_text = self.cfg.guide.text
        if not self.cfg.guide.append_direction:
            text_z = self.diffusion.get_text_embeds([ref_text])
            text_string = ref_text
        else:
            text_z = []
            text_string = []
            for d in self.view_dirs:
                text = ref_text.format(d)
                if d != 'front':
                    text = "" # JA: For all non-frontal views, we wish to use a null string prompt
                text_string.append(text)
                logger.info(text)
                negative_prompt = None
                logger.info(negative_prompt)
                text_z.append(self.diffusion.get_text_embeds([text], negative_prompt=negative_prompt))
        return text_z, text_string # JA: text_z contains the embedded vectors of the six view prompts

    def init_dataloaders(self) -> Dict[str, DataLoader]:
        if self.cfg.guide.use_zero123plus:
            init_train_dataloader = Zero123PlusDataset(self.cfg.render, device=self.device).dataloader()
        else:
            init_train_dataloader = MultiviewDataset(self.cfg.render, device=self.device).dataloader()

        val_loader = ViewsDataset(self.cfg.render, device=self.device,
                                  size=self.cfg.log.eval_size).dataloader()
        # Will be used for creating the final video
        val_large_loader = ViewsDataset(self.cfg.render, device=self.device,
                                        size=self.cfg.log.full_eval_size).dataloader()
        dataloaders = {'train': init_train_dataloader, 'val': val_loader,
                       'val_large': val_large_loader}
        return dataloaders

    def init_logger(self):
        logger.remove()  # Remove default logger
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format)
        logger.add(self.exp_path / 'log.txt', colorize=False, format=log_format)

    def paint(self):
        if self.cfg.guide.use_zero123plus:
            self.paint_zero123plus()
        else:
            self.paint_legacy()

    def paint_zero123plus(self):
        logger.info('Starting training ^_^')
        # Evaluate the initialization
        #MJ: self.evaluate(self.dataloaders['val'], self.eval_renders_path)
        self.mesh_model.train()

        viewpoint_data = []
        depths_rgba = []

        # JA: This color is the same as the background color of the image grid generated by Zero123++.
        background = torch.Tensor([0.5, 0.5, 0.5]).to(self.device)
        front_image = None
        outputs = None

        for i, data in enumerate(self.dataloaders['train']):
            if i == 0:
                # JA: The first viewpoint should always be frontal. It creates the extended version of the cropped
                # front view image. #MJ: self.paint_viewpoint() is modified to return   
                # outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius, background=background)
                #MJ rgb_output_front, object_mask_front = self.paint_viewpoint(data, should_project_back=False)
                outputs  = self.paint_viewpoint(data, should_project_back=False)
                # JA: The object mask is multiplied by the output to erase any generated part of the image that
                # "leaks" outside the boundary of the mesh from the front viewpoint. This operation turns the
                # background black, but we would like to use a white background, which is why we set the inverse 
                # of the mask to a ones tensor (the tensor is normalized to be between 0 and 1).
                rgb_output_front = outputs['image']
                object_mask_front =  outputs['mask']
                
                front_image = rgb_output_front * object_mask_front \
                    + torch.ones_like(rgb_output_front, device=self.device) * (1 - object_mask_front)
              
            else: 
            # JA: Even though the legacy function calls self.mesh_model.render for a similar purpose as for what
            # we do below, we still do the rendering again for the front viewpoint outside of the function for
            # the sake of brevity.

            # JA: Similar to what the original paint_viewpoint function does, we save all render outputs, that is,
            # the results from the renderer. In paint_viewpoint, rendering happened at the start of each viewpoint
            # and the images were generated using the depth/inpainting pipeline, one by one.

                theta, phi, radius = data['theta'], data['phi'], data['radius']
                phi = phi - np.deg2rad(self.cfg.render.front_offset)    # JA: The front azimuth angles of some meshes are
                                                                        # not 0. In that case, we need to add the front
                                                                        # azimuth offset
                phi = float(phi + 2 * np.pi if phi < 0 else phi) # JA: We convert negative phi angles to positive

                outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius, background=background) #MJ: Why not providing the multiple viewponits as a batch
                #MJ: render_cache = outputs['render_cache'] # JA: All the render outputs have the shape of (1200, 1200)

            #MJ: render again using the median color => The following results in media_color =[nan,nan,nan] => outputs is all nan
                # outputs = self.mesh_model.render(
                #     background=background,
                #     render_cache=render_cache,
                #     use_median=True
                # )
            #End   else
            #MJ: viewpoint_data is not used later
            # viewpoint_data.append({
            #     "render_outputs": outputs,
            #     "update_mask": outputs["mask"]
            # })

            # JA: In the depth controlled Zero123++ code example, the test depth map is found here:
            # https://d.skis.ltd/nrp/sample-data/0_depth.png
            # As it can be seen here, the foreground is closer to 0 (black) and background closer to 1 (white).
            # This is opposite of the SD 2.0 pipeline and the TEXTure internal renderer and must be inverted
            # (i.e. 1 minus the depth map, since the depth map is normalized to be between 0 and 1)
            depth = 1 - outputs['depth']
            mask = outputs['mask']

            # JA: The generated depth only has one channel, but the Zero123++ pipeline requires an RGBA image.
            # The mask is the object mask, such that the background has value of 0 and the foreground a value of 1.
            depth_rgba = torch.cat((depth, depth, depth, mask), dim=1)
            depths_rgba.append(depth_rgba)
            
        #MJ: for i, data in enumerate(self.dataloaders['train'])
        
        min_h, min_w, max_h, max_w = utils.get_nonzero_region(object_mask_front[0, 0])
        crop = lambda x: x[:, :, min_h:max_h, min_w:max_w]
        cropped_front_image = crop(front_image)

        zero123plus_cond = pad_tensor_to_size(cropped_front_image, 1200, 1200, value=1) # JA: pad the front view image with ones so that the resulting image will be 1200x1200. This makes the background white
                                                                                                # This is new code. cropped_front_image is the cropped version. In control zero123 pipeline, zero123_front_input is used without padding

        # JA: depths_rgba is a list that arranges the rows of the depth map, row by row
        # These depths are not cropped versions
        depth_grid = torch.cat((
            torch.cat((depths_rgba[1], depths_rgba[4]), dim=3),
            torch.cat((depths_rgba[2], depths_rgba[5]), dim=3),
            torch.cat((depths_rgba[3], depths_rgba[6]), dim=3),
        ), dim=2)

        # JA: From: https://pytorch.org/vision/main/generated/torchvision.transforms.ToPILImage.html
        # Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape H x W x C to a PIL Image
        # while adjusting the value range depending on the mode.
        # From: https://www.geeksforgeeks.org/python-pil-image-resize-method/
        # Parameters: 
        # size – The requested size in pixels, as a 2-tuple: (width, height).

        # JA: Zero123++ was trained with 320x320 images: https://github.com/SUDO-AI-3D/zero123plus/issues/70
        cond_image = torchvision.transforms.functional.to_pil_image(zero123plus_cond[0]).resize((320, 320))
        depth_image = torchvision.transforms.functional.to_pil_image(depth_grid[0]).resize((640, 960))

        @torch.enable_grad
        def on_step_end(pipeline, i, t, callback_kwargs): 
            #MJ: call_back_kwargs is created if the callback func, callback_on_step_end, is defined
            # as:  compute the previous noisy sample x_t -> x_t-1
            #    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]called.
            
            grid_latent = callback_kwargs["latents"] #MJ: The latent image being denoised; of shape (B,4,H,W)=(1,4,120,80); 80=320*2/8
            #MJ: grid_latent refers to the current latent that is denoised one step
            
            latents = split_zero123plus_grid(grid_latent, 320 // pipeline.vae_scale_factor)  #MJ: 320/8 = 40; tensor => a list of list of tensors
            blended_latents = []

            for viewpoint_index, data in enumerate(self.dataloaders['train']): #MJ: there are 7 viewpoints including the front view, 0.
                # if viewpoint_index == 0:
                #     continue  #MJ: do not use continue to implement the interleaving mode of SyncImg2Texture; 
                #     #  we need to project back the front view image which has been noisified at the same level 
                #     #  as the other view images being denoised by zero123plus

                theta, phi, radius = data['theta'], data['phi'], data['radius']
                phi = phi - np.deg2rad(self.cfg.render.front_offset)
                phi = float(phi + 2 * np.pi if phi < 0 else phi)

                #MJ: def render(self, theta=None, phi=None, radius=None, background=None,use_meta_texture=False, render_cache=None, use_median=False, dims=None, use_batch_render=True):
                outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius, background=background) #MJ: render using the texture being learned
                rgb_render_raw = outputs['image']

                render_cache = outputs['render_cache']  #MJ: render_cache is a dict
                #MJ: Render again to make the render image look better; It is not needed for zero123plus version
                # outputs = self.mesh_model.render(
                #     background=torch.Tensor([0.5, 0.5, 0.5]).to(self.device),
                #     render_cache=render_cache,
                #     use_median=True
                # )

                rgb_render = outputs['image']  
                #MJ: rgb_render is the output y of the rendering fuction whose parameters are the texture atlas;
                # So, rgb_render.requires_grad is TRUE

                #MJ: Get the view images being denoised; In the case of the front view image, it is already denoised;
                # We will noisify the front view image at the same level of noise as the other view images being denoised by the callback_step_end 
                # function of the zero123plus pipeline
                
                #MJ: added
                if viewpoint_index == 0:
                    #MJ: front_image is a full resolution (1200,1200) image in range of [0,1] => [-1,1]; handles Pil, ndarray, Tensor image
                    
                    front_image_for_zero123plus  =  \
                           F.interpolate(front_image, (320, 320), mode='nearest') #MJ: (1,3,1200,1200) => (1,3,320,320)
                    
                    preprocessed_front_image  = pipeline.image_processor.preprocess(front_image_for_zero123plus) 
                      #MJ:    preprocessed_front_image:  shape = (1,3,1200,1200) 
                    front_image_latent = self.scale_latents( 
                           pipeline.vae.encode(  #MJ: encode the rendered gt image: (B,3,H,W) => (B,4,H/8, W/8) => (1,4,40,40)
                           self.scale_image( preprocessed_front_image.half()),
                           return_dict=False
                           )[0].sample() * pipeline.vae.config.scaling_factor
                    )
                
                    noise = torch.randn_like( front_image_latent )
                    noised_front_image_latent = pipeline.scheduler.add_noise(front_image_latent, noise, t[None]) #MJ: t (starts from 999) is used to noisify gt_latents
                    latent = noised_front_image_latent  #MJ: latent: (1,4,40,40)
                else:
                    image_row_index = (viewpoint_index - 1) % 3
                    image_col_index = (viewpoint_index - 1) // 3

                    latent = latents[image_row_index][image_col_index]  #MJ: latent: a tensor of  shape (B,4,H,W) = (1,4,40,40); 40 = 320/8

                #MJ: Create the mask to mark the newly visited region on the mesh => We will adjust the current latent which has been
                # denoised one step, by blending it with the gt rendered image of the viewpoint
                
                diff = (rgb_render_raw.detach() - torch.tensor(self.mesh_model.default_color).view(1, 3, 1, 1).to(
                    self.device)).abs().sum(axis=1)
                exact_generate_mask = (diff < 0.1).float().unsqueeze(0)
                
                #MJ: diff < 0.1 means that the rendered area is influenced by the texture being learned rather the default, magenta color, texture color
                
                #MJ: What if we use the object mask, i.e., outputs['mask'], as curr_mask: => Test it: outputs['mask']  will be larger than exact_generate_mask
                #Note that in project_back, when we compute the rendering loss of the whole mesh, we use outputs['mask'] as the mask for each view
                #  Using outputs['mask'] makes sense when using the interveaving mode; In this mode, all the views are visited for each denoising step,
                #After all the views are visited once, diff will greater than 0.1 and so no region will be influenced by the generated image by the pipeline
                #But it does not makes sense. Also, using the object mask for blending means that the generated image from a view should be
                # aligned with the rendered image from that view. So the purpose of using the mask is simplified.  In the TEXTure,
                # the mask was used to distinguish the keep region, the generated region, and the refine region to maintain the consistency
                # between the views. In the interveaving mode of SyncImg2Texture, the texture regions of all the views are simultaneously 
                # generated, so we do not need to keep track, the keep, generate, refine regions. It is needed so that the texture generation in the later
                # views does nullify the previous generation of the texture atlas. In  the interveaving mode of SyncImg2Texture, there is no notion
                # of good previous generation of the texture; 
                
                # Extend mask
                generate_mask = torch.from_numpy(
                    cv2.dilate(exact_generate_mask[0, 0].detach().cpu().numpy(), np.ones((19, 19), np.uint8))).to(
                    exact_generate_mask.device).unsqueeze(0).unsqueeze(0)

                #MJ: downscale the generate_mask to the size of the latent space, because it is used to blend those images in the latent space
                curr_mask = F.interpolate(
                    generate_mask,
                    (320 // pipeline.vae_scale_factor, 320 // pipeline.vae_scale_factor),
                    mode='nearest'
                )

                rgb_render_small = F.interpolate(rgb_render, (320, 320), mode='bilinear', align_corners=False)

                #Moved out of the loop; Currently, in every loop the def statement itself defined. The inner functions are also hard to read
                # def scale_latents(latents): #MJ: move out of the loop
                #     latents = (latents - 0.22) * 0.75
                #     return latents

                # def unscale_latents(latents):
                #     latents = latents / 0.75 + 0.22
                #     return latents

                # def scale_image(image):
                #     image = image * 0.5 / 0.8
                #     return image

                # def unscale_image(image):
                #     image = image / 0.5 * 0.8
                #     return image

                # JA: When the generated latent tensor is denoised entirely, the Zero123++ pipeline uniquely
                # performs operations in the process of turning the latent space tensor z into pixel space
                # tensor x in the following manner:
                #   x = postprocess(unscale_image(vae_decode(unscale_latents(z) / scaling_factor)))
                # In order to move pixel space tensor x into latent space tensor z, the inverse must be
                # applied in the following manner:
                #   z = scale_latents(vae_encode(scale_image(preprocess(x))) * scaling_factor)

                preprocessed_rgb_render_small = pipeline.image_processor.preprocess(rgb_render_small) #MJ: rgb_render_small=nan;  rgb_render_small  (320,320)  in range of [0,1] => [-1,1]; handles Pil, ndarray, Tensor image

                gt_latents = self.scale_latents(
                    pipeline.vae.encode(  #MJ: encode the rendered gt image: (B,3,H,W) => (B,4,H/8, W/8)
                        self.scale_image(preprocessed_rgb_render_small.half()),
                        return_dict=False
                    )[0].sample() * pipeline.vae.config.scaling_factor
                )

                noise = torch.randn_like(gt_latents)
                noised_truth = pipeline.scheduler.add_noise(gt_latents, noise, t[None]) #MJ: t is used to noisify gt_latents; g_latents = nan

                
                # This blending equation is originally from TEXTure: The blending for each viewpoint is done within the viewpoint loop
                # Adjust the current latent which has been
                # denoised one step, by aligning it with the gt rendered image of the viewpoint
                
                blended_latent = latent * curr_mask + noised_truth * (1 - curr_mask) #MJ: noised_truth is nan at the first iteration i=0, t= 999
                #MJ: We can do the blending operation in a batch mode, having making latent and noised_truth have the batch size more than 1;
                # => We can eliminate the inefficient "for loop", which is executed every denoising step.
               
                blended_latents.append(blended_latent) # blended_latent = latent * curr_mask + noised_truth * (1 - curr_mask)
            #End for viewpoint_index, data in enumerate(self.dataloaders['train'])
            
            callback_kwargs["latents"] = torch.cat((
                torch.cat((blended_latents[0], blended_latents[3]), dim=3),
                torch.cat((blended_latents[1], blended_latents[4]), dim=3),
                torch.cat((blended_latents[2], blended_latents[5]), dim=3),
            ), dim=2).half()
            
            #MJ: callback_kwargs["latents"] will be used as the input to the next denoising step through the Unet
            
            #MJ: In the interleaving mode of SyncImg2Texture, call project_back here to learn the texture atlas from the view images
            # being denoised.
            
            
            
        #MJ: The following lines used to project back the denoised viewpoint images to the texture atlas; 
        # It is moved into callback_on_step_end function on_step_end:
        
            #grid_image = torchvision.transforms.functional.pil_to_tensor(result).to(self.device).float() / 255
            #MJ: grid_image is already avaiable as callback_kwargs["latents"]

            grid_latents = callback_kwargs["latents"]   #MJ: grid_latents.shape: torch.Size([1, 4, 120, 80])
            #MJ: We will project the curruntly denoised latent, grid_image;
            # But before that, we need to decode it, because project_back() requires the images in the pixel space

            #MJ: confer https://github.com/SUDO-AI-3D/zero123plus/blob/main/diffusers-support/pipeline.py
            
            #In Zero123Plus, after the latents are scaled, they are trained in the latent space;
            # So, after the denoising, they should be unscaled            
            unscaled_latents = self.unscale_latents(grid_latents)
            #In Zero123Plus, the images are also scaled for training. So, after denoising, they are unscaled
            decoded_unscaled_image = self.unscale_image(pipeline.vae.decode(unscaled_latents / pipeline.vae.config.scaling_factor, return_dict=False)[0])
            #MJ: Tensor decoded_unscaled_image ranges over [0,1]
    
            #  def postprocess(
            #     self,
            #     image: torch.FloatTensor,
            #     output_type: str = "pil",
            #     do_denormalize: Optional[List[bool]] = None,
            # ) -> Union[PIL.Image.Image, np.ndarray, torch.FloatTensor]
            #decoded_grid_image =  pipeline.image_processor.postprocess(unscaled_image.detach(), output_type="pil")
            #MJ: decoded_grid_image is a tensor of shape (1,3,960,640)             
            images = split_zero123plus_grid( decoded_unscaled_image, 320) #MJ: decoded_grid_image is tensor; images is a list of list of tensors

            thetas, phis, radii = [], [], []
            update_masks = []
            rgb_outputs = []

            for viewpoint, data in enumerate(self.dataloaders['train']):
                if viewpoint == 0:
                    image = front_image
                else:
                    image_row_index = (viewpoint - 1) % 3
                    image_col_index = (viewpoint - 1) // 3

                    image = images[image_row_index][image_col_index]
                    
                #MJ: Change the shape of the decoded image to the full size (1200,1200) of the rendered image
                #MJ: image =(1,3,1200,1200)
                rgb_output = F.interpolate(image, (1200, 1200), mode='bilinear', align_corners=False) #MJ: we can transform it as a batch mode
                rgb_outputs.append(rgb_output)

                theta, phi, radius = data['theta'], data['phi'], data['radius']
                phi = phi - np.deg2rad(self.cfg.render.front_offset)
                phi = float(phi + 2 * np.pi if phi < 0 else phi)

                thetas.append(theta)  #MJ: Create self.thetas in the init method to save computing time
                phis.append(phi)
                radii.append(radius)

                # JA: Create trimap of keep, refine, and generate using the render output
                #MJ: update_masks is not used; it is the same as object_mask = outputs['mask'] #
                # update_masks.append(viewpoint_data[i]["update_mask"])  
                
            #End for viewpoint, data in enumerate(self.dataloaders['train'])\
                
            outputs = self.mesh_model.render(theta=thetas, phi=phis, radius=radii, background=background)

            render_cache = outputs['render_cache'] # JA: All the render outputs have the shape of (1200, 1200)

           
            # Render again with the median value to use as rgb, we shouldn't have color leakage, but just in case
            #MJ: The following call is not used in zero123plus version
            # outputs = self.mesh_model.render(background=background,
            #                                render_cache=render_cache, use_median=True)

            # Render meta texture map
            #MJ: Meta_texture_img (the texture representing the z_normals of the whole mesh) is not used.
            #meta_output = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
            #                                    use_meta_texture=True, render_cache=render_cache)

            # JA: Get the Z component of the face normal vectors relative to the camera
            z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)  #MJ: z_normals can be computed as self.z_normals in the init method
            #z_normals_cache = meta_output['image'].clamp(0, 1) =>  z_normals_cache is not used
            z_normals_cache = None
            object_mask = outputs['mask'] # JA: mask has a shape of 1200x1200

            #MJ: project_back uses the full size image of rgb_outputs (1200x1200)
            self.project_back_only_texture_atlas(iter=i, #MJ: denoising iteration
                render_cache=render_cache, background=background, rgb_output=torch.cat(rgb_outputs),
                object_mask=object_mask, update_mask=object_mask, z_normals=z_normals, z_normals_cache=z_normals_cache
            )

            return callback_kwargs
        #End  def on_step_end(pipeline, i, t, callback_kwargs)
        
        # JA: Here we call the Zero123++ pipeline
        result = self.zero123plus(
            cond_image,
            depth_image=depth_image,
            num_inference_steps=36,
            callback_on_step_end=on_step_end
        ).images[0]

        #MJ: The following lines used to project back the denoised viewpoint images to the texture atlas; It
        # is moved into callback_on_step_end function on_step_end:
        
        # grid_image = torchvision.transforms.functional.pil_to_tensor(result).to(self.device).float() / 255

        # images = split_zero123plus_grid(grid_image, 320)

        # thetas, phis, radii = [], [], []
        # update_masks = []
        # rgb_outputs = []

        # for i, data in enumerate(self.dataloaders['train']):
        #     if i == 0:
        #         image = front_image
        #     else:
        #         image_row_index = (i - 1) % 3
        #         image_col_index = (i - 1) // 3

        #         image = images[image_row_index][image_col_index][None]

        #     rgb_output = F.interpolate(image, (1200, 1200), mode='bilinear', align_corners=False)
        #     rgb_outputs.append(rgb_output)

        #     theta, phi, radius = data['theta'], data['phi'], data['radius']
        #     phi = phi - np.deg2rad(self.cfg.render.front_offset)
        #     phi = float(phi + 2 * np.pi if phi < 0 else phi)

        #     thetas.append(theta)
        #     phis.append(phi)
        #     radii.append(radius)

        #     # JA: Create trimap of keep, refine, and generate using the render output
        #     update_masks.append(viewpoint_data[i]["update_mask"])

        # outputs = self.mesh_model.render(theta=thetas, phi=phis, radius=radii, background=background)

        # render_cache = outputs['render_cache'] # JA: All the render outputs have the shape of (1200, 1200)

        # # Render again with the median value to use as rgb, we shouldn't have color leakage, but just in case
        # outputs = self.mesh_model.render(background=background,
        #                                 render_cache=render_cache, use_median=True)

        # # Render meta texture map
        # meta_output = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
        #                                     use_meta_texture=True, render_cache=render_cache)

        # # JA: Get the Z component of the face normal vectors relative to the camera
        # z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
        # z_normals_cache = meta_output['image'].clamp(0, 1)
        # object_mask = outputs['mask'] # JA: mask has a shape of 1200x1200

        # self.project_back_only_texture_atlas(
        #     render_cache=render_cache, background=background, rgb_output=torch.cat(rgb_outputs),
        #     object_mask=object_mask, update_mask=object_mask, z_normals=z_normals, z_normals_cache=z_normals_cache
        # )

        self.mesh_model.change_default_to_median()
        logger.info('Finished Painting ^_^')
        logger.info('Saving the last result...')
        self.full_eval()
        logger.info('\tDone!')

    def paint_legacy(self):
        logger.info('Starting training ^_^')
        # Evaluate the initialization
        self.evaluate(self.dataloaders['val'], self.eval_renders_path)
        self.mesh_model.train()

        pbar = tqdm(total=len(self.dataloaders['train']), initial=self.paint_step,
                    bar_format='{desc}: {percentage:3.0f}% painting step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        # JA: The following loop computes the texture atlas for the given mesh using ten render images. In other words,
        # it is the inverse rendering process. Each of the ten views is one of the six view images.
        for data in self.dataloaders['train']:
            self.paint_step += 1
            pbar.update(1)
            self.paint_viewpoint(data) # JA: paint_viewpoint computes the part of the texture atlas by using a specific view image
            self.evaluate(self.dataloaders['val'], self.eval_renders_path)  # JA: This is the validation step for the current
                                                                            # training step
            self.mesh_model.train() # JA: Set the model to train mode because the self.evaluate sets the model to eval mode.

        self.mesh_model.change_default_to_median()
        logger.info('Finished Painting ^_^')
        logger.info('Saving the last result...')
        self.full_eval()
        logger.info('\tDone!')

    def evaluate(self, dataloader: DataLoader, save_path: Path, save_as_video: bool = False):
        logger.info(f'Evaluating and saving model, painting iteration #{self.paint_step}...')
        self.mesh_model.eval()
        save_path.mkdir(exist_ok=True)

        if save_as_video:
            all_preds = []
        for i, data in enumerate(dataloader):  #MJ: len(dataloader) = 10; This is the old setting
            preds, textures, depths, normals = self.eval_render(data)

            pred = tensor2numpy(preds[0])

            if save_as_video:
                all_preds.append(pred)
            else:
                Image.fromarray(pred).save(save_path / f"step_{self.paint_step:05d}_{i:04d}_rgb.jpg")
                Image.fromarray((cm.seismic(normals[0, 0].cpu().numpy())[:, :, :3] * 255).astype(np.uint8)).save(
                    save_path / f'{self.paint_step:04d}_{i:04d}_normals_cache.jpg')
                if self.paint_step == 0:
                    # Also save depths for debugging
                    torch.save(depths[0], save_path / f"{i:04d}_depth.pt")

        # Texture map is the same, so just take the last result
        texture = tensor2numpy(textures[0])
        Image.fromarray(texture).save(save_path / f"step_{self.paint_step:05d}_texture.png")

        if save_as_video:
            all_preds = np.stack(all_preds, axis=0)

            dump_vid = lambda video, name: imageio.mimsave(save_path / f"step_{self.paint_step:05d}_{name}.mp4", video,
                                                           fps=25,
                                                           quality=8, macro_block_size=1)

            dump_vid(all_preds, 'rgb')
        logger.info('Done!')

    def full_eval(self, output_dir: Path = None):
        if output_dir is None:
            output_dir = self.final_renders_path #MJ: = "results"
        self.evaluate(self.dataloaders['val_large'], output_dir, save_as_video=True)
        # except:
        #     logger.error('failed to save result video')

        if self.cfg.log.save_mesh:
            save_path = make_path(self.exp_path / 'mesh')
            logger.info(f"Saving mesh to {save_path}")

            self.mesh_model.export_mesh(save_path)

            logger.info(f"\tDone!")

    # JA: paint_viewpoint computes a portion of the texture atlas for the given viewpoint
    def paint_viewpoint(self, data: Dict[str, Any], should_project_back=True): #MJ: called with  should_project_back=False
        logger.info(f'--- Painting step #{self.paint_step} ---')
        theta, phi, radius = data['theta'], data['phi'], data['radius'] # JA: data represents a viewpoint which is stored in the dataset
        # If offset of phi was set from code
        phi = phi - np.deg2rad(self.cfg.render.front_offset)
        phi = float(phi + 2 * np.pi if phi < 0 else phi)
        logger.info(f'Painting from theta: {theta}, phi: {phi}, radius: {radius}')

        # Set background image
        if  True: #self.cfg.guide.second_model_type in ["zero123", "control_zero123"]: #self.view_dirs[data['dir']] != "front":
            # JA: For Zero123, the input image background is always white
            background = torch.Tensor([1, 1, 1]).to(self.device)
        elif self.cfg.guide.use_background_color: # JA: When use_background_color is True, set the background to the green color
            background = torch.Tensor([0, 0.8, 0]).to(self.device)
        else: # JA: Otherwise, set the background to the brick image
            background = F.interpolate(self.back_im.unsqueeze(0),
                                       (self.cfg.render.train_grid_size, self.cfg.render.train_grid_size),
                                       mode='bilinear', align_corners=False)

        # Render from viewpoint
        outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius, background=background, use_batch_render=False) #MJ: MJ ADD use_batch_render=False
        render_cache = outputs['render_cache'] # JA: All the render outputs have the shape of (1200, 1200)
        rgb_render_raw = outputs['image']  # Render where missing values have special color
        depth_render = outputs['depth']
        # Render again with the median value to use as rgb, we shouldn't have color leakage, but just in case
        #MJ: The following call is not used in zero123plus version
        #outputs = self.mesh_model.render(background=background,  #MJ: self.paint_step  refers to the viewpoint step
        #                                 render_cache=render_cache, use_median=self.paint_step > 1, use_batch_render=False)
        rgb_render = outputs['image']
        # Render meta texture map
        #MJ: meta_texture output is not used in zero123plus version
        # meta_output = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
        #                                      use_meta_texture=True, render_cache=render_cache, use_batch_render=False)

        z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
        #MJ: z_normals_cache = meta_output['image'].clamp(0, 1)
        #MJ: edited_mask = meta_output['image'].clamp(0, 1)[:, 1:2]
        z_normals_cache = None  #MJ: added by MJ
        edited_mask = None

        self.log_train_image(rgb_render, 'rendered_input')
        self.log_train_image(depth_render[0, 0], 'depth', colormap=True)
        self.log_train_image(z_normals[0, 0], 'z_normals', colormap=True)
        #MJ: self.log_train_image(z_normals_cache[0, 0], 'z_normals_cache', colormap=True)

        # text embeddings
        if self.cfg.guide.append_direction:
            dirs = data['dir']  # [B,]
            text_z = self.text_z[dirs] # JA: dirs is one of the six directions. text_z is the embedding vector of the specific view prompt
            text_string = self.text_string[dirs]
        else:
            text_z = self.text_z
            text_string = self.text_string
        logger.info(f'text: {text_string}')

        # JA: Create trimap of keep, refine, and generate using the render output
        
        update_mask, generate_mask, refine_mask = self.calculate_trimap(rgb_render_raw=rgb_render_raw,
                                                                        depth_render=depth_render,
                                                                        z_normals=z_normals,
                                                                        z_normals_cache=z_normals_cache,
                                                                        edited_mask=edited_mask,
                                                                        mask=outputs['mask'])

        update_ratio = float(update_mask.sum() / (update_mask.shape[2] * update_mask.shape[3]))
        if self.cfg.guide.reference_texture is not None and update_ratio < 0.01:
            logger.info(f'Update ratio {update_ratio:.5f} is small for an editing step, skipping')
            return

        self.log_train_image(rgb_render * (1 - update_mask), name='masked_input')
        self.log_train_image(rgb_render * refine_mask, name='refine_regions')

        # Crop to inner region based on object mask
        min_h, min_w, max_h, max_w = utils.get_nonzero_region(outputs['mask'][0, 0])
        
        crop = lambda x: x[:, :, min_h:max_h, min_w:max_w]  #MJ: select the object region from tensor x
        
        cropped_rgb_render = crop(rgb_render) # JA: This is rendered image which is denoted as Q_0.
                                              # In our experiment, 1200 is cropped to 827
        cropped_depth_render = crop(depth_render)
        cropped_update_mask = crop(update_mask)
       
        self.log_train_image(cropped_rgb_render, name='cropped_input')
        self.log_train_image(cropped_depth_render.repeat_interleave(3, dim=1), name='cropped_depth')

        checker_mask = None
        if self.paint_step > 1 or self.cfg.guide.initial_texture is not None:
            # JA: generate_checkerboard is defined in formula 2 of the paper
            checker_mask = self.generate_checkerboard(crop(update_mask), crop(refine_mask),
                                                      crop(generate_mask))
            self.log_train_image(F.interpolate(cropped_rgb_render, (512, 512)) * (1 - checker_mask),
                                 'checkerboard_input')
        self.diffusion.use_inpaint = self.cfg.guide.use_inpainting and self.paint_step > 1
        # JA: self.zero123_front_input has been added for Zero123 integration
        if self.zero123_front_input is None:
            resized_zero123_front_input = None
        else: # JA: Even though zero123 front input is fixed, it will be resized to the rendered image of each viewpoint other than the front view
            resized_zero123_front_input = F.interpolate(
                self.zero123_front_input,
                (cropped_rgb_render.shape[-2], cropped_rgb_render.shape[-1]) # JA: (H, W)
            )

        condition_guidance_scales = None
        if self.cfg.guide.individual_control_of_conditions:
            if self.cfg.guide.second_model_type != "control_zero123":
                raise NotImplementedError

            assert self.cfg.guide.guidance_scale_i is not None
            assert self.cfg.guide.guidance_scale_t is not None

            condition_guidance_scales = {
                "i": self.cfg.guide.guidance_scale_i,
                "t": self.cfg.guide.guidance_scale_t
            }

        # JA: Compute target image corresponding to the specific viewpoint, i.e. front, left, right etc. image
        # In the original implementation of TEXTure, the view direction information is contained in text_z. In
        # the new version, text_z 
        # D_t (depth map) = cropped_depth_render, Q_t (rendered image) = cropped_rgb_render.
        # Trimap is defined by update_mask and checker_mask. cropped_rgb_output refers to the result of the
        # Modified Diffusion Process.

        # JA: So far, the render image was created. Now we generate the image using the SD pipeline
        # Our pipeline uses the rendered image in the process of generating the image.
        cropped_rgb_output, steps_vis = self.diffusion.img2img_step(text_z, cropped_rgb_render.detach(), # JA: We use the cropped rgb output as the input for the depth pipeline
                                                                    cropped_depth_render.detach(),
                                                                    guidance_scale=self.cfg.guide.guidance_scale,
                                                                    strength=1.0, update_mask=cropped_update_mask,
                                                                    fixed_seed=self.cfg.optim.seed,
                                                                    check_mask=checker_mask,
                                                                    intermediate_vis=self.cfg.log.vis_diffusion_steps,

                                                                    # JA: The following were added to use the view image
                                                                    # created by Zero123
                                                                    view_dir=self.view_dirs[dirs], # JA: view_dir = "left", this is used to check if the view direction is front
                                                                    front_image=resized_zero123_front_input,
                                                                    phi=data['phi'],
                                                                    theta=data['base_theta'] - data['theta'],
                                                                    condition_guidance_scales=condition_guidance_scales)

        self.log_train_image(cropped_rgb_output, name='direct_output')
        self.log_diffusion_steps(steps_vis)
        # JA: cropped_rgb_output, as the output of sd pipeline, always has a shape of (512, 512); recover the resolution of the nonzero rendered image (e.g. (827, 827))
        cropped_rgb_output = F.interpolate(cropped_rgb_output, 
                                           (cropped_rgb_render.shape[2], cropped_rgb_render.shape[3]),
                                           mode='bilinear', align_corners=False)

        # Extend rgb_output to full image size
        # JA: After the image is generated, we insert it into the original RGB output
        rgb_output = rgb_render.clone() # JA: rgb_render shape is 1200x1200
        rgb_output[:, :, min_h:max_h, min_w:max_w] = cropped_rgb_output # JA: For example, (189, 1016, 68, 895) refers to the nonzero region of the render image
        self.log_train_image(rgb_output, name='full_output')

        # Project back
        object_mask = outputs['mask'] # JA: mask has a shape of 1200x1200
        # JA: Compute a part of the texture atlas corresponding to the target render image of the specific viewpoint
        
        if should_project_back:  #MJ: not used in zero123plus version
            fitted_pred_rgb, _ = self.project_back(render_cache=render_cache, background=background, rgb_output=rgb_output,
                                                object_mask=object_mask, update_mask=update_mask, z_normals=z_normals,
                                                z_normals_cache=z_normals_cache)
            self.log_train_image(fitted_pred_rgb, name='fitted')

        # JA: Zero123 needs the input image without the background
        # rgb_output is the generated and uncropped image in pixel space
        zero123_input = crop(
            rgb_output * object_mask
            + torch.ones_like(rgb_output, device=self.device) * (1 - object_mask)
        )   # JA: In the case of front view, the shape is (930,930).
            # This rendered image will be compressed to the shape of (512, 512) which is the shape of the diffusion
            # model.

        if self.view_dirs[dirs] == "front":
            self.zero123_front_input = zero123_input
        
        # if self.zero123_inputs is None:
        #     self.zero123_inputs = []
        
        # self.zero123_inputs.append({
        #     'image': zero123_input,
        #     'phi': data['phi'],
        #     'theta': data['theta']
        # })

        self.log_train_image(zero123_input, name='zero123_input')

        #MJ: return rgb_output, object_mask
        return outputs
    
    def eval_render(self, data):
        theta = data['theta']
        phi = data['phi']
        radius = data['radius']
        phi = phi - np.deg2rad(self.cfg.render.front_offset)
        phi = float(phi + 2 * np.pi if phi < 0 else phi)
        dim = self.cfg.render.eval_grid_size
        outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                         dims=(dim, dim), background='white')
        z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
        rgb_render = outputs['image']  # .permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        diff = (rgb_render.detach() - torch.tensor(self.mesh_model.default_color).view(1, 3, 1, 1).to(
            self.device)).abs().sum(axis=1)
        uncolored_mask = (diff < 0.1).float().unsqueeze(0)
        rgb_render = rgb_render * (1 - uncolored_mask) + utils.color_with_shade([0.85, 0.85, 0.85], z_normals=z_normals,
                                                                                light_coef=0.3) * uncolored_mask

        outputs_with_median = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                                     dims=(dim, dim), use_median=True,
                                                     render_cache=outputs['render_cache'])

        meta_output = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                             background=torch.Tensor([0, 0, 0]).to(self.device),
                                             use_meta_texture=True, render_cache=outputs['render_cache'])
        pred_z_normals = meta_output['image'][:, :1].detach()
        rgb_render = rgb_render.permute(0, 2, 3, 1).contiguous().clamp(0, 1).detach()
        texture_rgb = outputs_with_median['texture_map'].permute(0, 2, 3, 1).contiguous().clamp(0, 1).detach()
        depth_render = outputs['depth'].permute(0, 2, 3, 1).contiguous().detach()

        return rgb_render, texture_rgb, depth_render, pred_z_normals

    def calculate_trimap(self, rgb_render_raw: torch.Tensor,
                         depth_render: torch.Tensor,
                         z_normals: torch.Tensor, z_normals_cache: torch.Tensor, edited_mask: torch.Tensor,
                         mask: torch.Tensor):
        diff = (rgb_render_raw.detach() - torch.tensor(self.mesh_model.default_color).view(1, 3, 1, 1).to(
            self.device)).abs().sum(axis=1)
        exact_generate_mask = (diff < 0.1).float().unsqueeze(0)

        # Extend mask
        generate_mask = torch.from_numpy(
            cv2.dilate(exact_generate_mask[0, 0].detach().cpu().numpy(), np.ones((19, 19), np.uint8))).to(
            exact_generate_mask.device).unsqueeze(0).unsqueeze(0)

        update_mask = generate_mask.clone()

        object_mask = torch.ones_like(update_mask)
        object_mask[depth_render == 0] = 0
        object_mask = torch.from_numpy(
            cv2.erode(object_mask[0, 0].detach().cpu().numpy(), np.ones((7, 7), np.uint8))).to(
            object_mask.device).unsqueeze(0).unsqueeze(0)

        # Generate the refine mask based on the z normals, and the edited mask

        refine_mask = torch.zeros_like(update_mask)
        
        #MJ: we do not use z_normals_cache
        #refine_mask[z_normals > (z_normals_cache[:, :1, :, :] + self.cfg.guide.z_update_thr) ] = 1
        
        if self.cfg.guide.initial_texture is None:
            #MJ: we do not use z_normals_cache
            #refine_mask[z_normals_cache[:, :1, :, :] == 0] = 0
            pass
        elif self.cfg.guide.reference_texture is not None:
            refine_mask[edited_mask == 0] = 0
            refine_mask = torch.from_numpy(
                cv2.dilate(refine_mask[0, 0].detach().cpu().numpy(), np.ones((31, 31), np.uint8))).to(
                mask.device).unsqueeze(0).unsqueeze(0)
            refine_mask[mask == 0] = 0
            # Don't use bad angles here: It assumes that z_normals will be greater than 0.4 in other views
            refine_mask[z_normals < 0.4] = 0
        else:
            # Update all regions inside the object
            refine_mask[mask == 0] = 0

        refine_mask = torch.from_numpy(
            cv2.erode(refine_mask[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))).to(
            mask.device).unsqueeze(0).unsqueeze(0)
        refine_mask = torch.from_numpy(
            cv2.dilate(refine_mask[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))).to(
            mask.device).unsqueeze(0).unsqueeze(0)
        update_mask[refine_mask == 1] = 1

        update_mask[torch.bitwise_and(object_mask == 0, generate_mask == 0)] = 0

        # Visualize trimap
        if self.cfg.log.log_images:
            trimap_vis = utils.color_with_shade(color=[112 / 255.0, 173 / 255.0, 71 / 255.0], z_normals=z_normals)
            trimap_vis[mask.repeat(1, 3, 1, 1) == 0] = 1
            trimap_vis = trimap_vis * (1 - exact_generate_mask) + utils.color_with_shade(
                [255 / 255.0, 22 / 255.0, 67 / 255.0],
                z_normals=z_normals,
                light_coef=0.7) * exact_generate_mask

            shaded_rgb_vis = rgb_render_raw.detach()
            shaded_rgb_vis = shaded_rgb_vis * (1 - exact_generate_mask) + utils.color_with_shade([0.85, 0.85, 0.85],
                                                                                                 z_normals=z_normals,
                                                                                                 light_coef=0.7) * exact_generate_mask

            if self.paint_step > 1 or self.cfg.guide.initial_texture is not None:
                refinement_color_shaded = utils.color_with_shade(color=[91 / 255.0, 155 / 255.0, 213 / 255.0],
                                                                 z_normals=z_normals)
                only_old_mask_for_vis = torch.bitwise_and(refine_mask == 1, exact_generate_mask == 0).float().detach()
                trimap_vis = trimap_vis * 0 + 1.0 * (trimap_vis * (
                        1 - only_old_mask_for_vis) + refinement_color_shaded * only_old_mask_for_vis)
            self.log_train_image(shaded_rgb_vis, 'shaded_input')
            self.log_train_image(trimap_vis, 'trimap')

        return update_mask, generate_mask, refine_mask

    def generate_checkerboard(self, update_mask_inner, improve_z_mask_inner, update_mask_base_inner):
        checkerboard = torch.ones((1, 1, 64 // 2, 64 // 2)).to(self.device)
        # Create a checkerboard grid
        checkerboard[:, :, ::2, ::2] = 0
        checkerboard[:, :, 1::2, 1::2] = 0
        checkerboard = F.interpolate(checkerboard,
                                     (512, 512))
        checker_mask = F.interpolate(update_mask_inner, (512, 512))
        only_old_mask = F.interpolate(torch.bitwise_and(improve_z_mask_inner == 1,
                                                        update_mask_base_inner == 0).float(), (512, 512))
        checker_mask[only_old_mask == 1] = checkerboard[only_old_mask == 1]
        return checker_mask

    def project_back(self, render_cache: Dict[str, Any], background: Any, rgb_output: torch.Tensor,
                     object_mask: torch.Tensor, update_mask: torch.Tensor, z_normals: torch.Tensor,
                     z_normals_cache: torch.Tensor):
        eroded_masks = []
        for i in range(object_mask.shape[0]):  # Iterate over the batch dimension
            eroded_mask = cv2.erode(object_mask[i, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))
            eroded_masks.append(torch.from_numpy(eroded_mask).to(self.device).unsqueeze(0).unsqueeze(0))

        # Convert the list of tensors to a single tensor
        eroded_object_mask = torch.cat(eroded_masks, dim=0)
        # object_mask = torch.from_numpy(
        #     cv2.erode(object_mask[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))).to(
        #     object_mask.device).unsqueeze(0).unsqueeze(0)
        # render_update_mask = object_mask.clone()
        render_update_mask = eroded_object_mask.clone()

        # render_update_mask[update_mask == 0] = 0
        render_update_mask[update_mask == 0] = 0

        # blurred_render_update_mask = torch.from_numpy(
        #     cv2.dilate(render_update_mask[0, 0].detach().cpu().numpy(), np.ones((25, 25), np.uint8))).to(
        #     render_update_mask.device).unsqueeze(0).unsqueeze(0)
        dilated_masks = []
        for i in range(object_mask.shape[0]):  # Iterate over the batch dimension
            dilated_mask = cv2.dilate(render_update_mask[i, 0].detach().cpu().numpy(), np.ones((25, 25), np.uint8))
            dilated_masks.append(torch.from_numpy(dilated_mask).to(self.device).unsqueeze(0).unsqueeze(0))

        # Convert the list of tensors to a single tensor
        blurred_render_update_mask = torch.cat(dilated_masks, dim=0)
        blurred_render_update_mask = utils.gaussian_blur(blurred_render_update_mask, 21, 16)

        # Do not get out of the object
        blurred_render_update_mask[object_mask == 0] = 0

        if self.cfg.guide.strict_projection:
            blurred_render_update_mask[blurred_render_update_mask < 0.5] = 0
            # Do not use bad normals
            if z_normals is not None and z_normals_cache is not None:
                z_was_better = z_normals + self.cfg.guide.z_update_thr < z_normals_cache[:, :1, :, :]
                blurred_render_update_mask[z_was_better] = 0

        render_update_mask = blurred_render_update_mask
        for i in range(rgb_output.shape[0]):
            self.log_train_image(rgb_output[i][None] * render_update_mask[i][None], f'project_back_input_{i}')

        # Update the normals
        if z_normals is not None and z_normals_cache is not None:
            z_normals_cache[:, 0, :, :] = torch.max(z_normals_cache[:, 0, :, :], z_normals[:, 0, :, :])

        optimizer = torch.optim.Adam(self.mesh_model.get_params(), lr=self.cfg.optim.lr, betas=(0.9, 0.99),
                                     eps=1e-15)
            
        # JA: Create the texture atlas for the mesh using each view. It updates the parameters
        # of the neural network and the parameters are the pixel values of the texture atlas.
        # The loss function of the neural network is the render loss. The loss is the difference
        # between the specific image and the rendered image, rendered using the current estimate
        # of the texture atlas.
        # losses = []
        for _ in tqdm(range(200), desc='fitting mesh colors'):
            optimizer.zero_grad()
            outputs = self.mesh_model.render(background=background,
                                             render_cache=render_cache)
            rgb_render = outputs['image']

            mask = render_update_mask.flatten()
            masked_pred = rgb_render.reshape(1, rgb_render.shape[1], -1)[:, :, mask > 0]
            masked_target = rgb_output.reshape(1, rgb_output.shape[1], -1)[:, :, mask > 0]
            masked_mask = mask[mask > 0]
            loss = ((masked_pred - masked_target.detach()).pow(2) * masked_mask).mean()

            if z_normals is not None and z_normals_cache is not None:
                meta_outputs = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                                    use_meta_texture=True, render_cache=render_cache)
                current_z_normals = meta_outputs['image']
                current_z_mask = meta_outputs['mask'].flatten()
                masked_current_z_normals = current_z_normals.reshape(1, current_z_normals.shape[1], -1)[:, :,
                                        current_z_mask == 1][:, :1]
                masked_last_z_normals = z_normals_cache.reshape(1, z_normals_cache.shape[1], -1)[:, :,
                                        current_z_mask == 1][:, :1]
                loss += (masked_current_z_normals - masked_last_z_normals.detach()).pow(2).mean()
            # losses.append(loss.cpu().detach().numpy())
            loss.backward() # JA: Compute the gradient vector of the loss with respect to the trainable parameters of
                            # the network, that is, the pixel value of the texture atlas
            optimizer.step()

        if z_normals is not None and z_normals_cache is not None:
            return rgb_render, current_z_normals
        else:
            return rgb_render
        
    def project_back_only_texture_atlas(self, iter: int, render_cache: Dict[str, Any], background: Any, rgb_output: torch.Tensor,
                     object_mask: torch.Tensor, update_mask: torch.Tensor, z_normals: torch.Tensor,
                     z_normals_cache: torch.Tensor):
        eroded_masks = []
        for i in range(object_mask.shape[0]):  # Iterate over the batch dimension
            eroded_mask = cv2.erode(object_mask[i, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))
            eroded_masks.append(torch.from_numpy(eroded_mask).to(self.device).unsqueeze(0).unsqueeze(0))

        # Convert the list of tensors to a single tensor
        eroded_object_mask = torch.cat(eroded_masks, dim=0)
        render_update_mask = eroded_object_mask.clone()
        render_update_mask[update_mask == 0] = 0

        dilated_masks = []
        for i in range(object_mask.shape[0]):  # Iterate over the batch dimension
            dilated_mask = cv2.dilate(render_update_mask[i, 0].detach().cpu().numpy(), np.ones((25, 25), np.uint8))
            dilated_masks.append(torch.from_numpy(dilated_mask).to(self.device).unsqueeze(0).unsqueeze(0))

        # Convert the list of tensors to a single tensor
        blurred_render_update_mask = torch.cat(dilated_masks, dim=0)
        blurred_render_update_mask = utils.gaussian_blur(blurred_render_update_mask, 21, 16)

        # Do not get out of the object
        blurred_render_update_mask[object_mask == 0] = 0

        if self.cfg.guide.strict_projection:
            blurred_render_update_mask[blurred_render_update_mask < 0.5] = 0

        render_update_mask = blurred_render_update_mask
        for i in range(rgb_output.shape[0]):
            self.log_train_image(rgb_output[i][None] * render_update_mask[i][None], f'project_back_input_{i}')

        optimizer = torch.optim.Adam(self.mesh_model.get_params(), lr=self.cfg.optim.lr, betas=(0.9, 0.99),
                                     eps=1e-15)
            
        # JA: Create the texture atlas for the mesh using each view. It updates the parameters
        # of the neural network and the parameters are the pixel values of the texture atlas.
        # The loss function of the neural network is the render loss. The loss is the difference
        # between the specific image and the rendered image, rendered using the current estimate
        # of the texture atlas.
        # losses = []
        with tqdm( range(self.cfg.optim.epochs),desc='fitting mesh colors') as pbar: #MJ: epochs = 100
            for _ in pbar:
                optimizer.zero_grad()
                outputs = self.mesh_model.render(background=background,
                                                render_cache=render_cache)
                rgb_render = outputs['image']

                # loss = (render_update_mask * (rgb_render - rgb_output.detach()).pow(2)).mean()
                loss = (render_update_mask * z_normals * (rgb_render - rgb_output.detach()).pow(2)).mean()

                loss.backward() # JA: Compute the gradient vector of the loss with respect to the trainable parameters of
                                # the network, that is, the pixel value of the texture atlas
                optimizer.step()
                
                #MJ:  Update the description of the progress bar with current loss
                pbar.set_description(f"Fitting mesh colors -At Iter ={iter}, Epoch {_ + 1}, Loss: {loss.item():.4f}")

        return rgb_render

    def log_train_image(self, tensor: torch.Tensor, name: str, colormap=False):
        if self.cfg.log.log_images:
            if colormap:
                tensor = cm.seismic(tensor.detach().cpu().numpy())[:, :, :3]
            else:
                tensor = einops.rearrange(tensor, '(1) c h w -> h w c').detach().cpu().numpy()
            Image.fromarray((tensor * 255).astype(np.uint8)).save(
                self.train_renders_path / f'{self.paint_step:04d}_{name}.jpg')

    def log_diffusion_steps(self, intermediate_vis: List[Image.Image]):
        if len(intermediate_vis) > 0:
            step_folder = self.train_renders_path / f'{self.paint_step:04d}_diffusion_steps'
            step_folder.mkdir(exist_ok=True)
            for k, intermedia_res in enumerate(intermediate_vis):
                intermedia_res.save(
                    step_folder / f'{k:02d}_diffusion_step.jpg')

    def save_image(self, tensor: torch.Tensor, path: Path):
        if self.cfg.log.log_images:
            Image.fromarray(
                (einops.rearrange(tensor, '(1) c h w -> h w c').detach().cpu().numpy() * 255).astype(np.uint8)).save(
                path)

