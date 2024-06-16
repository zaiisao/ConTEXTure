import time
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

from torch_scatter import scatter_max

import torchvision
from PIL import Image
from diffusers import DiffusionPipeline, ControlNetModel

from src import utils
from src.configs.train_config import TrainConfig
from src.models.textured_mesh import TexturedMeshModel
from src.stable_diffusion_depth import StableDiffusion
from src.training.views_dataset import Zero123PlusDataset, ViewsDataset, MultiviewDataset
from src.utils import make_path, tensor2numpy, pad_tensor_to_size, split_zero123plus_grid

from PIL import Image, ImageDraw
# JA: scale_latents, unscale_latents, scale_image, and unscale_image are from the Zero123++ pipeline code:
# https://huggingface.co/sudo-ai/zero123plus-pipeline/blob/main/pipeline.py
def scale_latents(latents):
    latents = (latents - 0.22) * 0.75
    return latents

def unscale_latents(latents):
    latents = latents / 0.75 + 0.22
    return latents

def scale_image(image):
    image = image * 0.5 / 0.8
    return image

def unscale_image(image):
    image = image / 0.5 * 0.8
    return image

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
        
       
    def create_face_view_map(self, face_idx):
        num_views, _, H, W = face_idx.shape  # Assume face_idx shape is (B, 1, H, W)

        # Flatten the face_idx tensor to make it easier to work with
        face_idx_flattened_2d = face_idx.view(num_views, -1)  # Shape becomes (num_views, H*W)

        # Get the indices of all elements
        # JA: From ChatGPT:
        # torch.meshgrid is used to create a grid of indices that corresponds to each dimension of the input tensor,
        # specifically in this context for the view indices and pixel indices. It allows us to pair each view index
        # with every pixel index, thereby creating a full coordinate system that can be mapped directly to the values
        # in the tensor face_idx.
        view_by_pixel_indices, pixel_by_view_indices = torch.meshgrid(
            torch.arange(num_views, device=face_idx.device),
            torch.arange(H * W, device=face_idx.device),
            indexing='ij'
        )

        # Flatten indices tensors
        view_by_pixel_indices_flattened = view_by_pixel_indices.flatten()
        pixel_by_view_indices_flattened = pixel_by_view_indices.flatten()

        faces_idx_view_pixel_flattened = face_idx_flattened_2d.flatten()

        # Convert pixel indices back to 2D indices (i, j)
        pixel_i_indices = pixel_by_view_indices_flattened // W
        pixel_j_indices = pixel_by_view_indices_flattened % W

        # JA: The original face view map is made of nested dictionaries, which is very inefficient. Face map information
        # is implemented as a single tensor which is efficient. Only tensors can be processed in GPU; dictionaries cannot
        # be processed in GPU.
        # The combined tensor represents, for each pixel (i, j), its view_idx 
        combined_tensor_for_face_view_map = torch.stack([
            faces_idx_view_pixel_flattened,
            view_by_pixel_indices_flattened,
            pixel_i_indices,
            pixel_j_indices
        ], dim=1)

        # Filter valid faces
        faces_idx_valid_mask = faces_idx_view_pixel_flattened >= 0

        # JA:
        # [[face_id_1, view_1, i_1, j_1]
        #  [face_id_1, view_1, i_2, j_2]
        #  [face_id_1, view_1, i_3, j_3]
        #  [face_id_1, view_2, i_4, j_4]
        #  [face_id_1, view_2, i_5, j_5]
        #  ...
        #  [face_id_2, view_1, i_k, j_l]
        #  [face_id_2, view_1, i_{k + 1}, j_{l + 1}]
        #  [face_id_2, view_2, i_{k + 2}, j_{l + 2}]]
        #  ...
        # The above example shows face_id_1 is projected, under view_1, to three pixels (i_1, j_1), (i_2, j_2), (i_3, j_3)
        # Shape is Nx4 where N is the number of pixels (no greater than H*W*num_views = 1200*1200*7) that projects the
        # valid face ID.
        return combined_tensor_for_face_view_map[faces_idx_valid_mask]

    def compare_face_normals_between_views(self,face_view_map, face_normals, face_idx):
        num_views, _, H, W = face_idx.shape
        weight_masks = torch.full((num_views, 1, H, W), True, dtype=torch.bool, device=face_idx.device)

        face_ids = face_view_map[:, 0] # JA: face_view_map.shape = (H*W*num_views, 4) = (1200*1200*7, 4) = (10080000, 4)
        views = face_view_map[:, 1]
        i_coords = face_view_map[:, 2]
        j_coords = face_view_map[:, 3]
        z_normals = face_normals[views, 2, face_ids] # JA: The shape of face_normals is (num_views, 3, num_faces)
                                                     # For example, face_normals can be (7, 3, 14232)
                                                     # z_normals is (N,)

        # Scatter z-normals into the tensor, ensuring each index only keeps the max value
        # JA: z_normals is the source/input tensor, and face_ids is the index tensor to scatter_max function.
        max_z_normals_over_views, _ = scatter_max(z_normals, face_ids, dim=0) # JA: N is a subset of length H*W*num_views
        # The shape of max_z_normals_over_N is the (num_faces,). The shape of the scatter_max output is equal to the
        # shape of the number of distinct indices in the index tensor face_ids.

        # Map the gathered max normals back to the respective face ID indices
        # JA: max_z_normals_over_views represents the max z normals over views for every face ID.
        # The shape of face_ids is (N,). Therefore the shape of max_z_normals_over_views_per_face is also (N,).
        max_z_normals_over_views_per_face = max_z_normals_over_views[face_ids]

        # Calculate the unworthy mask where current z-normals are less than the max per face ID
        unworthy_pixels_mask = z_normals < max_z_normals_over_views_per_face

        # JA: Update the weight masks. The shapes of face_view_map, whence views, i_coords, and j_coords were extracted
        # from, all have the shape of (N,), which represents the number of valid pixel entries. Therefore,
        # weight_masks[views, 0, i_coords, j_coords] will also have the shape of (N,) which allows the values in
        # weight_masks to be set in an elementwise manner.
        #
        # weight_masks[views[0], 0, i_coords[0], j_coords[0]] = ~(unworthy_pixels_mask[0])
        # The above variable represents whether the pixel (i_coords[0], j_coords[0]) under views[0] is worthy to
        # contribute to the texture atlas.
        weight_masks[views, 0, i_coords, j_coords] = ~(unworthy_pixels_mask)

        # weight_masks[views[0], 0, i_coords[0], j_coords[0]] = ~(unworthy_pixels_mask[0])
        # weight_masks[views[1], 0, i_coords[1], j_coords[1]] = ~(unworthy_pixels_mask[1])
        # weight_masks[views[2], 0, i_coords[2], j_coords[2]] = ~(unworthy_pixels_mask[2])
        # weight_masks[views[3], 0, i_coords[3], j_coords[2]] = ~(unworthy_pixels_mask[3])

        return weight_masks

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
            "sudo-ai/zero123plus-v1.1", custom_pipeline="src/zero123plus.py",
            torch_dtype=torch.float16
        )

        pipeline.add_controlnet(ControlNetModel.from_pretrained(
            "sudo-ai/controlnet-zp11-depth-v1", torch_dtype=torch.float16
        ), conditioning_scale=2)

        pipeline.to(self.device)

        pipeline.inpaint_unet = self.diffusion.inpaint_unet

        return pipeline

    def calc_text_embeddings(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        ref_text = self.cfg.guide.text
        if self.cfg.guide.use_zero123plus:
            assert not self.cfg.guide.append_direction, "append_direction should be False when use_zero123plus is True"

            text_z = []
            text_string = []

            text_string.append(ref_text)
            text_string.append(ref_text + ", front view")
            
            for text in text_string:
                negative_prompt = None
                text_z.append(self.diffusion.get_text_embeds([text], negative_prompt=negative_prompt))
        elif not self.cfg.guide.append_direction:
            text_z = self.diffusion.get_text_embeds([ref_text])
            text_string = ref_text
        else:
            text_z = []
            text_string = []
            for d in self.view_dirs:
                text = ref_text.format(d)
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

    def define_view_weights(self):
        # Set the camera poses:
        self.thetas = []
        self.phis = []
        self.radii = []
       
        for i, data in enumerate(self.dataloaders['train']):
            theta, phi, radius = data['theta'], data['phi'], data['radius']
            phi = phi - np.deg2rad(self.cfg.render.front_offset)
            phi = float(phi + 2 * np.pi if phi < 0 else phi)

            self.thetas.append(theta)
            self.phis.append(phi)
            self.radii.append(radius)

        if not self.cfg.optim.learn_max_z_normals:
            augmented_vertices = self.mesh_model.mesh.vertices

            batch_size = len(self.dataloaders['train'])

            # JA: We need to repeat several tensors to support the batch size.
            # For example, with a tensor of the shape, [1, 3, 1200, 1200], doing
            # repeat(batch_size, 1, 1, 1) results in [1 * batch_size, 3 * 1, 1200 * 1, 1200 * 1]
            _, _, _, face_normals, face_idx = self.mesh_model.render_face_normals_face_idx(
                augmented_vertices[None].repeat(batch_size, 1, 1),
                self.mesh_model.mesh.faces, # JA: the faces tensor can be shared across the batch and does not require its own batch dimension.
                self.mesh_model.face_attributes.repeat(batch_size, 1, 1, 1),
                elev=torch.tensor(self.thetas).to(self.device), # MJ: elev, azim, and radius should be tensors
                azim=torch.tensor(self.phis).to(self.device),
                radius=torch.tensor(self.radii).to(self.device),
                
                look_at_height=self.mesh_model.dy,
                background_type='none'
            )
            
            logger.info(f'Generating face view map')

            #MJ: get the binary masks for each view which indicates how much the image rendered from each view
            # should contribute to the texture atlas over the mesh which is the cause of the image
            face_view_map = self.create_face_view_map(face_idx)

            # logger.info(f'Creating weight masks for each view')
            weight_masks = self.compare_face_normals_between_views(face_view_map, face_normals, face_idx)

            self.view_weights = weight_masks
        else: 
            #MJ: you need to learn max_z_normals if you use learn_max_z_normals  to define self.view_weights
            self.project_back_max_z_normals()

            #MJ: Render all views using self.meta_texture_img (max_z_normals) learned  by  self.project_back_max_z_normals()           
            meta_outputs = self.mesh_model.render(theta=self.thetas, phi=self.phis, radius=self.radii,
                                                    background=torch.Tensor([0, 0, 0]).to(self.device),
                                                    use_meta_texture=True, render_cache=None)
            z_normals = meta_outputs["normals"][:,2:3,:,:].clamp(0, 1)
            max_z_normals = meta_outputs['image'][:,0:1,:,:].clamp(0, 1) 

            #MJ: max_z_normals refers to the projection of self.meta_texture_img, which is a leaf tensor (parameter tensor)
            self.view_weights = self.compute_view_weights(z_normals, max_z_normals, alpha=self.cfg.optim.alpha) #MJ: = -50 , -100, -10000
            # self.view_weights is a function of self.meta_texture_img; When self.view_weights is used to compute a loss
            # self.view_weights.detach() should be used to avoid an error of re-using the freed computational graph
    
            #MJ: for debugging:
            # max_z_normals_red = meta_outputs['image'][:,:,:,:].clamp(0, 1)       
            # for i in range(len(self.thetas)):
            #     self.log_train_image(
            #         torch.cat((z_normals[i][None], z_normals[i][None], z_normals[i][None]), dim=1),
            #         f'z_normals_{i}'
            #     )
            #     self.log_train_image(torch.cat((max_z_normals[i][None], max_z_normals[i][None],
            #                                        max_z_normals[i][None]), dim=1), f'max_z_normals_gray_{i}')
                
            #     self.log_train_image(max_z_normals_red[i][None], f'max_z_normals_red_{i}')
                
            #     self.log_train_image(torch.cat((self.view_weights[i][None], self.view_weights[i][None], self.view_weights[i][None]), dim=1),
            #                           f'view_weights_{i}')  # MJ: view_weights: (B,1,H,W)
          
        # End of self.cfg.optim.learn_max_z_normals

        logger.info(f'Successfully initialized {self.cfg.log.exp_name}')

    def get_cropped_rgb_renders(self, rgb_renders, object_masks):
        B, _, _, _ = object_masks.shape
        cropped_rgb_renders_list = []
        for i in range(B):
            mask_i = object_masks[i, 0]
            min_h, min_w, max_h, max_w = utils.get_nonzero_region_tuple(mask_i) #MJ: outputs["mask"][0, 0]: shape (1,1,H,W)
            crop = lambda x: x[:, :, min_h:max_h, min_w:max_w]
            cropped_rgb_render = crop(rgb_renders[i][None])
            cropped_rgb_renders_list.append(cropped_rgb_render)

        return cropped_rgb_renders_list

    def paint_zero123plus(self):
        logger.info('Starting training ^_^')
        
        zero123_start_time = time.perf_counter()  # Record the start time
        zero123plus_prep_start_time = time.perf_counter()  # Record the end time
        
        self.define_view_weights()
        # Evaluate the initialization: Currently self.texture_img and self.meta_texture_img are not learned at all;
        #MJ:  Because we are not learning the textures sequentially scanning over the viewpoints, this evaluation is meaningless
        #MJ: self.evaluate(self.dataloaders['val'], self.eval_renders_path)
        self.mesh_model.train()

        # JA: This color is the same as the background color of the image grid generated by Zero123++.
        background_gray = torch.Tensor([0.5, 0.5, 0.5]).to(self.device)

        frontview_data_iter = iter(self.dataloaders['train'])
        frontview_data = next(frontview_data_iter)  # Gets the first batch
                        
        front_view_start_time = time.perf_counter()  # Record the start time
        rgb_output_front, object_mask_front = self.paint_viewpoint(frontview_data, should_project_back=True)
        # MJ: At this point, self.texture_img has been learned from the front view image; 
        # So when the mesh is rendered hereafter this learned texture map will be used.
        
        front_view_end_time = time.perf_counter()  # Record the end time
        front_view_elapsed_time = front_view_end_time - front_view_start_time  # Calculate elapsed time

        print(f"Elapsed time in Front view image generation with project-back: {front_view_elapsed_time} seconds")

        #MJ: At this point, self.texture_img has been learned from the front view image; 
        # So when the mesh is rendered, this learned texture map will be used to render the mesh   
        outputs_magenta = self.mesh_model.render(theta=self.thetas, phi=self.phis, radius=self.radii, background=background_gray)
        rgb_renders = outputs_magenta['image']
        render_cache = outputs_magenta['render_cache']
        object_masks = outputs_magenta['mask']

        B, _, _, _ = object_masks.shape

        cropped_rgb_renders_magenta_list = self.get_cropped_rgb_renders(rgb_renders, object_masks)
        masks_latent_list = []
    
        for i in range(B):
            # MJ: In a new viewpoint, when the image rendered using the partly learned texture atlas 
            # may contain the part which is rendered from the part of the texture atlas with the
            # initial magenta color. If so, it means that for this part, the image should be generated by the generation
            # pipeline. That is, if the newly rendered image is not different from the default texture color, 
            # it is taken to mean that the new region should be painted by the generated image.isidentifier()
            # But, the new region should be well-aligned with the part that is already valid; This part is
            # considered as the background in the context of inpainting.
                            
            #diff should be computed with respect to the cropped rgb render box itself, because the ground truth render image
            #(the object itself) is confined to that box
            cropped_diff = (cropped_rgb_renders_magenta_list[i].detach() - torch.tensor(self.mesh_model.default_color).view(1, 3, 1, 1).to(
                    self.device)).abs().sum(axis=1) # shape (1,909 909) with v=1
            cropped_generate_mask = (cropped_diff < 0.1).float().unsqueeze(0)
            
            # Dilate the bounary of the mask
            dilated_cropped_mask = torch.from_numpy(
                cv2.dilate(cropped_generate_mask[0, 0].detach().cpu().numpy(), np.ones((19, 19), np.uint8))).to(
                cropped_generate_mask.device).unsqueeze(0).unsqueeze(0)

            tile_size = 320 // self.zero123plus.vae_scale_factor #MJ: 320/8 = 40
            mask_latent = F.interpolate(
                 dilated_cropped_mask,
                (tile_size, tile_size),
                mode='nearest'  #MJ: 320 // pipeline.vae_scale_factor = 40
            )
            masks_latent_list.append( mask_latent)
        #End for i in range(B) 

        #MJ: Rendering the mesh again using use_median = True updates the default color
        # region (magenta color) of texture map with the median color of the learned texture map from the front view
        outputs_median = self.mesh_model.render(
            background=background_gray,
            render_cache=render_cache,
            use_median=True  
        )

        rgb_renders_median = outputs_median['image']
        cropped_rgb_renders_median_list = self.get_cropped_rgb_renders(rgb_renders_median, object_masks)

        # MJ: prepare the front view image (rendered using median-color of the texture map)
        rgb_output_front =   rgb_renders_median[0][None]
     
        front_image_rgba = torch.cat((rgb_output_front, object_mask_front), dim=1)
        min_h, min_w, max_h, max_w = utils.get_nonzero_region_tuple(object_mask_front[0, 0])
        crop = lambda x: x[:, :, min_h:max_h, min_w:max_w]
        cropped_front_image_rgba = crop(front_image_rgba)

        self.log_train_image(front_image_rgba[:, 0:3], 'paint_zero123plus:front_image (use_median)')
        self.log_train_image(cropped_front_image_rgba[:, 0:3], 'paint_zero123plus:cropped_front_image (use_median)')

        # JA: In the depth controlled Zero123++ code example, the test depth map is found here:
        # https://d.skis.ltd/nrp/sample-data/0_depth.png
        # As it can be seen here, the foreground is closer to 0 (black) and background closer to 1 (white).
        # This is opposite of the SD 2.0 pipeline and the TEXTure internal renderer and must be inverted
        # (i.e. 1 minus the depth map, since the depth map is normalized to be between 0 and 1)
        depth = 1 - outputs_median['depth']

        # JA: The generated depth only has one channel, but the Zero123++ pipeline requires an RGBA image.
        # The mask is the object mask, such that the background has value of 0 and the foreground a value of 1.
        # MJ: the mask is used as the alpha; background alpha = 0 means that it is transparent
        depth_rgba = torch.cat((depth, depth, depth, object_masks), dim=1)
        #MJ: depth_rgba.shape: torch.Size([7, 4, 1200, 1200])

        bounding_boxes = torch.zeros((B, 4), dtype=torch.int32)  # Prepare output tensor
        cropped_rgb_depths_small_list = []
        cropped_rgb_renders_small_list = []

        for i in range(B):
            mask_i = object_masks[i, 0] #MJ: object_masks: shape=(7,1,1200,1200); mask_i: shape = (1200,1200)
            bbox_i = utils.get_nonzero_region_tensor(mask_i)
            
            bounding_boxes[i, :] = bbox_i

            min_h, min_w, max_h, max_w = bbox_i
            crop = lambda x: x[:, :, min_h:max_h, min_w:max_w]
            cropped_depth_rgba = crop(depth_rgba[i][None])
            cropped_rgb_depths_small_list.append(cropped_depth_rgba)

            cropped_rgb_depths_small_list[i] = F.interpolate(
                cropped_rgb_depths_small_list[i], (320, 320),
                mode='bilinear',
                align_corners=False
            )
            
            cropped_rgb_renders_small_list.append(F.interpolate(
                cropped_rgb_renders_median_list[i], (320, 320),   
                mode='bilinear',
                align_corners=False
            ))

        # MJ: cropped_rgb_depths_small_list is a list of tensors 
        # JA: cropped_depths_rgba is a list that arranges the rows of the depth map, row by row
        cropped_depth_grid = torch.cat((
            torch.cat((cropped_rgb_depths_small_list[1], cropped_rgb_depths_small_list[4]), dim=3),
            torch.cat((cropped_rgb_depths_small_list[2], cropped_rgb_depths_small_list[5]), dim=3),
            torch.cat((cropped_rgb_depths_small_list[3], cropped_rgb_depths_small_list[6]), dim=3),
        ), dim=2)

        cropped_rgb_renders_grid = torch.cat((
            torch.cat((cropped_rgb_renders_small_list[1], cropped_rgb_renders_small_list[4]), dim=3),
            torch.cat((cropped_rgb_renders_small_list[2], cropped_rgb_renders_small_list[5]), dim=3),
            torch.cat((cropped_rgb_renders_small_list[3], cropped_rgb_renders_small_list[6]), dim=3),
        ), dim=2)

        masks_grid = torch.cat((
            torch.cat((masks_latent_list[1], masks_latent_list[4]), dim=3),
            torch.cat((masks_latent_list[2], masks_latent_list[5]), dim=3),
            torch.cat((masks_latent_list[3], masks_latent_list[6]), dim=3),
        ), dim=2)

        # JA: the Zero123++ pipeline converts the latent space tensor z into pixel space
        # tensor x in the following manner:
        #   x = postprocess(unscale_image(vae_decode(unscale_latents(z) / scaling_factor)))
        # It implies that  to convert pixel space tensor x into latent space tensor z, 
        # the inverse operation must be applied in the following manner:
        #   z = scale_latents(vae_encode(scale_image(preprocess(x))) * scaling_factor)
        preprocessed_cropped_rgb_renders_grid = self.zero123plus.image_processor.preprocess(cropped_rgb_renders_grid)
        scaled_cropped_rgb_renders_grid = scale_image(preprocessed_cropped_rgb_renders_grid.half())

        scaled_latent_renders_grid = self.zero123plus.vae.encode(  #MJ: encoding 320x320 to 40x40 by compressing 8 times
            scaled_cropped_rgb_renders_grid,
            return_dict=False
        )[0].sample() * self.zero123plus.vae.config.scaling_factor 

        gt_renders_latent_grid = scale_latents(scaled_latent_renders_grid)

        self.log_train_image(cropped_depth_grid[:, 0:3], 'zero123plus_prep:cropped_depth_grid')
        self.log_train_image(cropped_rgb_renders_grid, 'zero123plus_prep:cropped_rgb_renders_grid')
      
        # JA: From: https://pytorch.org/vision/main/generated/torchvision.transforms.ToPILImage.html
        # Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape H x W x C to a PIL Image
        # while adjusting the value range depending on the mode.
        # From: https://www.geeksforgeeks.org/python-pil-image-resize-method/
        # Parameters: 
        # size – The requested size in pixels, as a 2-tuple: (width, height).

        # JA: Zero123++ was trained with 320x320 images: https://github.com/SUDO-AI-3D/zero123plus/issues/70
        cond_image = torchvision.transforms.functional.to_pil_image(cropped_front_image_rgba[0]).resize((320, 320))
        depth_image = torchvision.transforms.functional.to_pil_image(cropped_depth_grid[0]).resize((640, 960))

        zero123plus_prep_end_time = time.perf_counter()  
        elapsed_time = zero123plus_prep_end_time - zero123plus_prep_start_time
        print(f'zero123plus_prep_time={elapsed_time:0.4f}')

        self.previous_grid_latent = None
        self.zero123plus_unet = self.zero123plus.unet

        self.should_inpaint = True

        zero123_start_time = time.perf_counter()  # Record the end time

        masked_input_latents = gt_renders_latent_grid * (masks_grid < 0.5) + 0.5 * (masks_grid >= 0.5)
        
        #MJ: Generate a 3x2 grid image conditioned on the front view image and the 6 depth maps of the mesh
        result = self.zero123plus(
            cond_image,
            prompt=self.text_string[0],
            depth_image=depth_image,
            num_inference_steps=50,

            use_inpaint=False,
            use_blending=True,
            latent_mask_grid=masks_grid.half(),
            latent_renders_grid=gt_renders_latent_grid,
            masked_input_latents=masked_input_latents.half()
        ).images[0]
        #MJ: return image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        zero123_end_time = time.perf_counter()  # Record the end time
        elapsed_time = zero123_end_time - zero123_start_time # Calculate elapsed time

        print(f"Elapsed time in zero123 sampling: {elapsed_time} seconds")

        #MJ: Now that the images for the 7 views are generated, project back them to construct the texture atlas
        project_back_prep_start_time =  time.perf_counter()  # Record the end time

        grid_image = torchvision.transforms.functional.pil_to_tensor(result).to(self.device).float() / 255

        self.log_train_image(grid_image[None], f'zero123plus:result_grid_image_{self.cfg.optim.seed}')

        images = split_zero123plus_grid(grid_image, 320)
        rgb_outputs = []
        for i in range(B):
            if i == 0:
                rgb_output = rgb_output_front
            else:
                image_row_index = (i - 1) % 3
                image_col_index = (i - 1) // 3

                rgb_output_small = images[image_row_index][image_col_index][None]

                # JA: Since Zero123++ requires cond tensor and each depth tensor to be of size 320x320, we resize this
                # to match what it used to be prior to scaling down.
                min_h, min_w, max_h, max_w = bounding_boxes[i]

                original_cropped_rgb_output = F.interpolate(
                    rgb_output_small,
                    (max_h - min_h, max_w - min_w),
                    mode='bilinear',
                    align_corners=False
                )

                # JA: We initialize rgb_output, the image where cropped_rgb_output will be "pasted into." Since the
                # renderer produces tensors (depth maps, object mask, etc.) with a height and width of 1200, rgb_output
                # is initialized with the same size so that it aligns pixel-wise with the renderer-produced tensors.
                # Because Zero123++ generates non-transparent images, that is, images without an alpha channel, with
                # a background of rgb(0.5, 0.5, 0.5), we initialize the tensor using torch.ones and multiply by 0.5.
                rgb_output_full = torch.ones(
                     original_cropped_rgb_output.shape[0],  original_cropped_rgb_output.shape[1], 1200, 1200
                ).to(self.device) * 0.5

                rgb_output_full[:, :, min_h:max_h, min_w:max_w] =  original_cropped_rgb_output

                rgb_output = F.interpolate(rgb_output_full, (1200, 1200), mode='bilinear', align_corners=False)
            #End else:
            rgb_outputs.append(rgb_output)
        #End for i in range(B)

        #MJ: Project-back the generated view images, rgb_outputs, to the texture atlas.
        project_back_prep_end_time = time.perf_counter()  # Record the end time
        
        elapsed_time = project_back_prep_end_time - project_back_prep_start_time
        print(f'project-back prep time={elapsed_time:0.4f}')

        project_back_start_time = time.perf_counter()  # Record the end time
        self.project_back_only_texture_atlas(   #When render_cache is not None, the camera pose is not needed
            render_cache=render_cache, background=background_gray, rgb_output=torch.cat(rgb_outputs),
            object_mask=object_masks, update_mask=object_masks, z_normals=None, z_normals_cache=None
        )

        project_back_end_time = time.perf_counter()  # Record the end time 

        elapsed_time = project_back_end_time - project_back_start_time
        print(f'project-back time={elapsed_time:0.4f}')

        self.mesh_model.change_default_to_median()
        logger.info('Finished Painting ^_^')
        logger.info('Saving the last result...')

        zero123_end_time = time.perf_counter()  # Record the end time
        total_elapsed_time = zero123_end_time - zero123_start_time  # Calculate elapsed time

        print(f"Total Elapsed time with zero123plus: {total_elapsed_time:.4f} seconds")

        #MJ: Render the mesh using the learned texture atlas from a lot of regularly sampled viewpoints
        # and create a video from the rendered image sequence and 
        self.full_eval() 
        logger.info('\t All Done!')

    def paint_legacy(self):
        logger.info('Starting training ^_^')
        
        TEXTure_start_time = time.perf_counter()  # Record the start time
        # Evaluate the initialization
        #MJ: self.evaluate(self.dataloaders['val'], self.eval_renders_path)
        self.mesh_model.train()

        pbar = tqdm(total=len(self.dataloaders['train']), initial=self.paint_step,
                    bar_format='{desc}: {percentage:3.0f}% painting step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        # JA: The following loop computes the texture atlas for the given mesh using ten render images. In other words,
        # it is the inverse rendering process. Each of the ten views is one of the six view images.
        for data in self.dataloaders['train']:
            self.paint_step += 1
            pbar.update(1)
            self.paint_viewpoint(data) # JA: paint_viewpoint computes the part of the texture atlas by using a specific view image
            #MJ: self.evaluate(self.dataloaders['val'], self.eval_renders_path)  # JA: This is the validation of the currently leared texture map using various camera viewpoints                                                                          # training step
            self.mesh_model.train() # JA: Set the model to train mode because the self.evaluate sets the model to eval mode.

        self.mesh_model.change_default_to_median()
        logger.info('Finished Painting ^_^')
        logger.info('Saving the last result...')
        
        
        TEXTure_end_time = time.perf_counter()  # Record the end  time
        total_elapsed_time =   TEXTure_end_time  -  TEXTure_start_time
        print(f"Total Elapsed time with TEXTure: {total_elapsed_time:.4f} seconds")

        self.full_eval()
        logger.info('\tDone!')

    def evaluate(self, dataloader: DataLoader, save_path: Path, save_as_video: bool = False): #MJ: dataloader=self.dataloaders['val']
        logger.info(f'Evaluating and saving model, painting iteration #{self.paint_step}...')
        self.mesh_model.eval()
        save_path.mkdir(exist_ok=True)

        if save_as_video: 
            all_preds = []
        for i, data in enumerate(dataloader):
            preds, textures, depths, normals = self.eval_render(data) #MJ: preds, textures, depths, normals = rgb_render, texture_rgb, depth_render, pred_z_normals
            #MJ: normals =  pred_z_normals = meta_output['image'][:, :1].detach() #MJ: pred_z_normals refers to max_z_normals
            pred = tensor2numpy(preds[0])

            if save_as_video:
                all_preds.append(pred)
            else:
                Image.fromarray(pred).save(save_path / f"eval:rendered_image:{i:04d}_rgb.jpg")
                Image.fromarray((cm.seismic(normals[0, 0].cpu().numpy())[:, :, :3] * 255).astype(np.uint8)).save(
                    save_path / f'eval:normal_map:{i:04d}_normals_cache.jpg')
                if self.paint_step == 0:
                    # Also save depths for debugging
                    torch.save(depths[0], save_path / f"eval:depth_map:{i:04d}_depth.pt")

        # Texture map is the same, so just take the last result
        texture = tensor2numpy(textures[0])
        Image.fromarray(texture).save(save_path / f"eval:texture_atlas:texture.png")
        
       
        
        
          
        if save_as_video:  #np.cat: Shape Change: If the input arrays have shape (A, B, C), the concatenated array will have shape (NA, B, C) if axis=0 (where N is the number of arrays).
            all_preds = np.stack(all_preds, axis=0) # combine a sequence of arrays along a new axis:  If the input arrays have shape (A, B, C), the stacked array will have shape (N, A, B, C) if axis=0 (where N is the number of arrays).

            
            dump_vid = lambda video, name: imageio.mimsave(save_path / f"eval:constructed_video:{name}_{self.cfg.optim.seed}.mp4", video,
                                                           fps=25,
                                                           quality=8, macro_block_size=1)

            dump_vid(all_preds, 'all_rendered_rgb')
        logger.info('Eval Done!')

    def full_eval(self, output_dir: Path = None):

        if output_dir is None:
            output_dir = self.final_renders_path
        self.evaluate(self.dataloaders['val_large'], output_dir, save_as_video=True)
        # except:
        #     logger.error('failed to save result video')

        if self.cfg.log.save_mesh:
            save_path = make_path(self.exp_path / 'mesh')
            logger.info(f"Saving mesh to {save_path}")

            self.mesh_model.export_mesh(save_path)

            logger.info(f"\t Full Eval Done!")

    # JA: paint_viewpoint computes a portion of the texture atlas for the given viewpoint
    def paint_viewpoint(self, data: Dict[str, Any], should_project_back=True):
        logger.info(f'--- Painting step #{self.paint_step} ---')
        theta, phi, radius = data['theta'], data['phi'], data['radius'] # JA: data represents a viewpoint which is stored in the dataset
        # If offset of phi was set from code
        phi = phi - np.deg2rad(self.cfg.render.front_offset)
        phi = float(phi + 2 * np.pi if phi < 0 else phi)
        logger.info(f'Painting from theta: {theta}, phi: {phi}, radius: {radius}')

        # Set background image
        if  self.cfg.guide.second_model_type in ["zero123", "control_zero123"]: #self.view_dirs[data['dir']] != "front":
            # JA: For Zero123, the input image background is always white
            background = torch.Tensor([1, 1, 1]).to(self.device)
        elif self.cfg.guide.use_background_color: # JA: When use_background_color is True, set the background to the green color
            background = torch.Tensor([0, 0.8, 0]).to(self.device)
        else: # JA: Otherwise, set the background to the brick image
            background = F.interpolate(self.back_im.unsqueeze(0),
                                       (self.cfg.render.train_grid_size, self.cfg.render.train_grid_size),
                                       mode='bilinear', align_corners=False)

        # Render from viewpoint
        outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius, background=background)
        render_cache = outputs['render_cache'] # JA: All the render outputs have the shape of (1200, 1200)
        rgb_render_raw = outputs['image']  #MJ: The rendered image without using use-median = True 
        depth_render = outputs['depth']
        
        # Render again with the median value to use as rgb, we shouldn't have color leakage, but just in case
        
     
       
        outputs = self.mesh_model.render(background=background,
                                          render_cache=render_cache, use_median=self.paint_step > 1)
        rgb_render = outputs['image']
        
        meta_output = self.mesh_model.render(background=background,
                                            use_meta_texture=True, render_cache=render_cache)

        z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
        z_normals_cache = meta_output['image'].clamp(0, 1)
        edited_mask = meta_output['image'].clamp(0, 1)[:, 1:2]

          
        self.log_train_image(rgb_render, 'paint_viewpoint:rgb_render')
        self.log_train_image(depth_render[0, 0], 'paint_viewpoint:depth', colormap=True)
        # self.log_train_image(z_normals[0, 0], 'paint_viewpoint:z_normals', colormap=True)
        # self.log_train_image(z_normals_cache[0, 0], 'paint_viewpoint:z_normals_cache', colormap=True)

        # text embeddings
        if self.cfg.guide.use_zero123plus:
            text_z = self.text_z[1]
            text_string = self.text_string[1]
            view_dir = "front"
        elif self.cfg.guide.append_direction:
            dirs = data['dir']  # [B,]
            text_z = self.text_z[dirs] # JA: dirs is one of the six directions. text_z is the embedding vector of the specific view prompt
            text_string = self.text_string[dirs]
            view_dir = self.view_dirs[dirs]
        else:
            text_z = self.text_z
            text_string = self.text_string
            view_dir = None
        logger.info(f'text: {text_string}')

        #MJ: Because we do not consider each viewpoint in a sequence to project-back the view image to the texture-atlas, we do not use the trimap
       
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

        self.log_train_image(rgb_render * (1 - update_mask), name='paint_viewpoint:masked_rgb_render')
        self.log_train_image(rgb_render * refine_mask, name='paint_viewpoint:refine_rgb_render')
        self.log_train_image( torch.cat([ refine_mask, refine_mask, refine_mask], dim=1), name='paint_viewpoint:refine_mask')
        self.log_train_image( torch.cat([ update_mask, update_mask, update_mask], dim=1), name='paint_viewpoint:update_mask') 
        self.log_train_image( torch.cat([ generate_mask, generate_mask, generate_mask], dim=1), name='paint_viewpoint:generate_mask')
       
        # Crop to inner region based on object mask
        # Crop to inner region based on object mask
        min_h, min_w, max_h, max_w = utils.get_nonzero_region_tuple(outputs['mask'][0, 0])
        crop = lambda x: x[:, :, min_h:max_h, min_w:max_w]
        cropped_rgb_render = crop(rgb_render) # JA: This is rendered image which is denoted as Q_0.
                                              # In our experiment, 1200 is cropped to 827
        cropped_depth_render = crop(depth_render)
        
        
        cropped_update_mask = crop(update_mask)
     
        self.log_train_image(cropped_rgb_render, name='paint_viewpoint:cropped_rgb_render')
        self.log_train_image(cropped_depth_render.repeat_interleave(3, dim=1), name='paint_viewpoint:cropped_depth')

        checker_mask = None
             
        if self.paint_step > 1 or self.cfg.guide.initial_texture is not None:
            # JA: generate_checkerboard is defined in formula 2 of the paper
            checker_mask = self.generate_checkerboard(crop(update_mask), crop(refine_mask),
                                                    crop(generate_mask))
            self.log_train_image(F.interpolate(cropped_rgb_render, (512, 512)) * (1 - checker_mask),
                                'checkerboard_input')
       
        
        start_time = time.perf_counter()  # Record the start time
        #rgb_output_front, object_mask_front = self.paint_viewpoint(data, should_project_back=True)
        
        #MJ: Use cropped_update_mask (which is the object mask) as the input to img2img_step, instead of the cropped rgb render image
        cropped_update_mask_rgb = torch.cat( [cropped_update_mask,cropped_update_mask,cropped_update_mask ],dim=1)
        # cropped_rgb_output, steps_vis = self.diffusion.img2img_step(text_z, #cropped_update_mask_rgb.detach(), 
        #                                                             cropped_rgb_render.detach(), # JA: We use the cropped rgb output as the input for the depth pipeline
        #                                                             cropped_depth_render.detach(),
        #                                                             guidance_scale=self.cfg.guide.guidance_scale,
        #                                                             strength=1.0, update_mask=cropped_update_mask,
        #                                                             fixed_seed=self.cfg.optim.seed,
        #                                                             check_mask=checker_mask,
        #                                                             intermediate_vis=self.cfg.log.vis_diffusion_steps,

        #                                                             # JA: The following were added to use the view image
        #                                                             # created by Zero123
        #                                                             view_dir=view_dir, # JA: view_dir = "left",e.g.; this is used to check if the view direction is front
        #                                                             front_image=resized_zero123_front_input,
        #                                                             phi=data['phi'],
        #                                                             theta=data['base_theta'] - data['theta'],
        #                                                             condition_guidance_scales=condition_guidance_scales)
        
        self.diffusion.use_inpaint = self.cfg.guide.use_inpainting and self.paint_step > 1
        cropped_rgb_output, steps_vis = self.diffusion.img2img_step(text_z, cropped_rgb_render.detach(),
                                                                    cropped_depth_render.detach(),
                                                                    guidance_scale=self.cfg.guide.guidance_scale,
                                                                    strength=1.0, update_mask=cropped_update_mask,
                                                                    fixed_seed=self.cfg.optim.seed,
                                                                    check_mask=checker_mask,
                                                                    intermediate_vis=self.cfg.log.vis_diffusion_steps)
        

        
        end_time = time.perf_counter()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate elapsed time

        print(f"Elapsed time in self.diffusion.img2img_step in TEXTureWithZero123: {elapsed_time:0.4f} seconds")
        
        self.log_train_image(cropped_rgb_output, name='paint_viewpoint:cropped_rgb_output (result of img2img) (magenta boundary?)')
        self.log_diffusion_steps(steps_vis)
        # JA: cropped_rgb_output always has a shape of (512, 512); recover the resolution of the nonzero rendered image (e.g. (827, 827))
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
        if should_project_back:
            if not self.cfg.guide.use_zero123plus:  
               fitted_pred_rgb = self.project_back(render_cache=render_cache, background=background, rgb_output=rgb_output,
                                                object_mask=object_mask, update_mask=update_mask,  z_normals=z_normals,
                                                z_normals_cache=z_normals_cache
                                                )
            else:
               fitted_pred_rgb = self.project_back(render_cache=render_cache, background=background, rgb_output=rgb_output,
                                                object_mask=object_mask, update_mask=update_mask,  z_normals=None,
                                                z_normals_cache=None
                                                )                                                                                  
            self.log_train_image(fitted_pred_rgb, name='paint_viewpoint:fitted_pred_rgb rendered using the texture map learned from the front view image')
            
            

        # JA: Zero123 needs the input image without the background
        # rgb_output is the generated and uncropped image in pixel space
        zero123_input = crop(
            rgb_output * object_mask
            + torch.ones_like(rgb_output, device=self.device) * (1 - object_mask)
        )   # JA: In the case of front view, the shape is (930,930).
            # This rendered image will be compressed to the shape of (512, 512) which is the shape of the diffusion
            # model.

        if view_dir == "front":
            self.zero123_front_input = zero123_input
        
        # if self.zero123_inputs is None:
        #     self.zero123_inputs = []
        
        # self.zero123_inputs.append({
        #     'image': zero123_input,
        #     'phi': data['phi'],
        #     'theta': data['theta']
        # })

        self.log_train_image(zero123_input, name='paint_viewpoint:zero123_cond_image')

        return rgb_output, object_mask

    def eval_render(self, data):
        theta = data['theta']
        phi = data['phi']
        radius = data['radius']
        phi = phi - np.deg2rad(self.cfg.render.front_offset)
        phi = float(phi + 2 * np.pi if phi < 0 else phi)
        dim = self.cfg.render.eval_grid_size
        
        #Now, self.texture_img has been learned fully (when we call eval_render even when self.texture_img is partially learned)
        outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                         dims=(dim, dim), background='white')
        
        
        z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
        rgb_render = outputs['image']  # .permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        
        #rgb_render.register_hook(self.print_hook) #MJ: for debugging with loss.backward(retrain_graph=True)
        
        diff = (rgb_render.detach() - torch.tensor(self.mesh_model.default_color).view(1, 3, 1, 1).to(
            self.device)).abs().sum(axis=1)
        uncolored_mask = (diff < 0.1).float().unsqueeze(0)
        rgb_render = rgb_render * (1 - uncolored_mask) + utils.color_with_shade([0.85, 0.85, 0.85], z_normals=z_normals,
                                                                                light_coef=0.3) * uncolored_mask
        #MJ: In case when  self.texture_img is not learned (still with the default magenta color), 
        # fill that with the mean color of the learned part
        outputs_with_median = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                                     dims=(dim, dim), #MJ: use_median=True,
                                                     render_cache=outputs['render_cache'])

        meta_output = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                             background=torch.Tensor([0, 0, 0]).to(self.device),
                                             use_meta_texture=True, render_cache=outputs['render_cache'])
        
        pred_z_normals = meta_output['image'][:, :1].detach() #MJ: pred_z_normals refers to max_z_normals
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

        # Extend exact_generate_mas mask
        generate_mask = torch.from_numpy(
            cv2.dilate(exact_generate_mask[0, 0].detach().cpu().numpy(), np.ones((19, 19), np.uint8))).to(
            exact_generate_mask.device).unsqueeze(0).unsqueeze(0)

        update_mask = generate_mask.clone()

        object_mask = torch.ones_like(update_mask)
        object_mask[depth_render == 0] = 0  #depth_render == 0 refers the background,  the non-object part: The background part is not be updated
        object_mask = torch.from_numpy(
            cv2.erode(object_mask[0, 0].detach().cpu().numpy(), np.ones((7, 7), np.uint8))).to(
            object_mask.device).unsqueeze(0).unsqueeze(0)

        # Generate the refine mask based on the z normals, and the edited mask

        refine_mask = torch.zeros_like(update_mask)
        refine_mask[z_normals > ( z_normals_cache[:, :1, :, :] + self.cfg.guide.z_update_thr ) ] = 1 #MJ: The part to be refined is the part where z_normals is greater than the max_z_normals_cache (==0 initially) +  0.2 
        if self.cfg.guide.initial_texture is None:
            refine_mask[z_normals_cache[:, :1, :, :] == 0] = 0  #MJ: refine_mask is zero for the front view, where z_normals_cache[:, :1, :, :] ==0
        elif self.cfg.guide.reference_texture is not None:
            refine_mask[edited_mask == 0] = 0
            refine_mask = torch.from_numpy(
                cv2.dilate(refine_mask[0, 0].detach().cpu().numpy(), np.ones((31, 31), np.uint8))).to(
                mask.device).unsqueeze(0).unsqueeze(0)
            refine_mask[mask == 0] = 0
            # Don't use bad angles here
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
        update_mask[refine_mask == 1] = 1  #MJ: Among the pixels of the update_mask, the part to be refined is set to 1``

        update_mask[torch.bitwise_and(object_mask == 0, generate_mask == 0)] = 0 #MJ: The non object part and the non-generation part is not updated

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
    
    #MJ: project_back() used only for the front view image, so we do not need to learn the meta-texture-img which holds
    #    the max_z_normals

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

        
        if not self.cfg.guide.use_zero123plus:
            if self.cfg.guide.strict_projection:
                blurred_render_update_mask[blurred_render_update_mask < 0.5] = 0
                # Do not use bad normals
                if z_normals is not None and z_normals_cache is not None:
                    z_was_better = z_normals + self.cfg.guide.z_update_thr < z_normals_cache[:, :1, :, :]
                    blurred_render_update_mask[z_was_better] = 0

        render_update_mask = blurred_render_update_mask
        for i in range(rgb_output.shape[0]):
            self.log_train_image(rgb_output[i][None] * render_update_mask[i][None], f'project_back:input_{i}')

        # Update the normals:
        
        if not self.cfg.guide.use_zero123plus:
        
            if z_normals is not None and z_normals_cache is not None:
                z_normals_cache[:, 0, :, :] = torch.max(z_normals_cache[:, 0, :, :], z_normals[:, 0, :, :])
                   
        else:
            pass
           
            # In ConTEXTure, project_back() used only for the front view image, 
            # so we do not need to learn the meta-texture-img which holds the max_z_normals
        if not self.cfg.guide.use_zero123plus:    
            optimizer = torch.optim.Adam(self.mesh_model.get_params(), lr=self.cfg.optim.lr,
                          betas=(0.9, 0.99),  eps=1e-15)
            
        else:      
           
            optimizer = torch.optim.Adam(self.mesh_model.get_params_texture_atlas(), lr=self.cfg.optim.lr, 
                                         betas=(0.9, 0.99),  eps=1e-15)
            
         
        # JA: Create the texture atlas for the mesh using each view. It updates the parameters
        # of the neural network and the parameters are the pixel values of the texture atlas.
        # The loss function of the neural network is the render loss. The loss is the difference
        # between the specific image and the rendered image, rendered using the current estimate
        # of the texture atlas.
        # losses = []
        with  tqdm(range(300), desc='project_back (SD2): fitting mesh colors for the front view') as pbar:
          for iter in pbar:  #MJ: Here we do not have the batch loop, but only the epoch loop
            optimizer.zero_grad() #MJ: This effectively resets the gradients before each backward pass;
            #  it resets these .grad attributes to zero for all parameters before the backward pass.
            #   preventing them from accumulating across iterations. 
            #   this avoids the accumulation of the gradients (which is the default behavior)
            #MJ: In PyTorch, gradients accumulate by default. When you call backward() multiple times, 
            # the gradients are added to the existing gradients in the x.grad attribute unless you manually zero them out.
            outputs = self.mesh_model.render(background=background, #MJ:  render_caches contains the info about all viewpoint projection and rasterization data
                                             render_cache=render_cache)
            rgb_render = outputs['image']

            mask = render_update_mask.flatten()
            masked_pred = rgb_render.reshape(1, rgb_render.shape[1], -1)[:, :, mask > 0]
            masked_target = rgb_output.reshape(1, rgb_output.shape[1], -1)[:, :, mask > 0]
            masked_mask = mask[mask > 0]
            loss = ((masked_pred - masked_target.detach()).pow(2) * masked_mask).mean()

              
            if not self.cfg.guide.use_zero123plus:
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
            else: 
                pass
                #MJ: project_back() used only for the front view image, so we do not need to learn the meta-texture-img
                # which holds the max_z_normals
            
            
            
            pbar.set_description(f"project_back (SD2): Fitting mesh colors -Epoch {iter}, Loss: {loss.item():.7f}")
            #MJ:    # Backward pass (accumulate gradients): cf: ChatGPT: https://chatgpt.com/share/12b5eaf3-eb97-425f-9142-78603e682683
            #  the gradient of the loss over a large batch is the sum of the gradients of the losses over the mini-batches.
            loss.backward() # JA: Compute the gradient vector of the loss with respect to the trainable parameters of
                            # the network, that is, the pixel value of the texture atlas
            optimizer.step()

                
        #End with  tqdm(range(200), desc='project_back (SD2): fitting mesh colors for the front view') as pbar:
        return rgb_render
        
    #   self.project_back_only_texture_atlas(
    #         render_cache=render_cache, background=background, rgb_output=torch.cat(rgb_outputs),
    #         object_mask=object_mask, update_mask=object_mask, z_normals=z_normals, z_normals_cache=z_normals_cache,
    #         weight_masks=self.weight_masks
    #     )
    
     
    def print_hook(self, grad):
           print(f"Gradient: {grad}")  
                 
    def project_back_only_texture_atlas(self, render_cache: Dict[str, Any], background: Any, rgb_output: torch.Tensor,
                     object_mask: torch.Tensor, update_mask: torch.Tensor, z_normals: torch.Tensor,
                      z_normals_cache: torch.Tensor                   
                     ):
        eroded_masks = []
        
        #object_mask = object_mask.detach()
        #object_mask is a part of the computational graph self.mesh_model.render()
        #This computational graph will be freed after loss.backward() in the training loop later.
        #MJ: render_update_mask used within the training loop may be freed because it is also a part of the computional graph of loss
        #Then, when render_update_mask is used at the second iteration of the loop, it will be no longer available.
        #Detaching object_mask from the computational graph will avoid the problem.
        
        for i in range(object_mask.shape[0]):  # Iterate over the batch dimension
            eroded_mask = cv2.erode(object_mask[i, 0].cpu().numpy(), np.ones((5, 5), np.uint8))
            #tensor.cpu(): Moves the tensor to the CPU. This does not affect the computational graph that the tensor is part of.
            
            #eroded_mask = cv2.erode(object_mask[i, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))
            eroded_masks.append(torch.from_numpy(eroded_mask).to(self.device).unsqueeze(0).unsqueeze(0))

        # Convert the list of tensors to a single tensor
        eroded_object_mask = torch.cat(eroded_masks, dim=0)
        render_update_mask = eroded_object_mask.clone()
        render_update_mask[update_mask == 0] = 0

        dilated_masks = []
        for i in range(object_mask.shape[0]):  # Iterate over the batch dimension
            dilated_mask = cv2.dilate(render_update_mask[i, 0].cpu().numpy(), np.ones((25, 25), np.uint8))
            #dilated_mask = cv2.dilate(render_update_mask[i, 0].detach().cpu().numpy(), np.ones((25, 25), np.uint8))
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
           
        optimizer = torch.optim.Adam(self.mesh_model.get_params_texture_atlas(), lr=self.cfg.optim.lr, betas=(0.9, 0.99),
                                     eps=1e-15)
        
            
        # JA: Create the texture atlas for the mesh using each view. It updates the parameters
        # of the neural network and the parameters are the pixel values of the texture atlas.
        # The loss function of the neural network is the render loss. The loss is the difference
        # between the specific image and the rendered image, rendered using the current estimate
        # of the texture atlas.
        # losses = []

        # JA: TODO: Add num_epochs hyperparameter
        with tqdm(range(300), desc='project_back_only_texture_atla: fitting mesh colors') as pbar:
            for iter in pbar:
                optimizer.zero_grad()
                
                outputs = self.mesh_model.render(background=background,
                                                render_cache=render_cache)  
                #MJ: with render_cache given (not None) => render() omits the raterization process,
                # uses the cache of the info about the projection and rasteriation of all viewpoints.
                #  and only performs the texture mapping projection => take much less time than performing a full rendering 
                                                   
                rgb_render = outputs['image']
                # # Debugging: Check the grad_fn attribute
                # print(f"Iteration {iter}: rgb_render.grad_fn: {rgb_render.grad_fn}")

                #rgb_render.register_hook(self.print_hook)
                
                # for i in range(rgb_render.shape[0]):
                #     self.log_train_image(rgb_render[i][None] * render_update_mask[i][None], f'project_back: rgb-output_{i}')  
           
               
                #MJ:  By calling detach(), you prevent gradients from flowing back into self.view_weights
                #  from this particular loss computation. This means self.view_weights can 
                #  still be updated elsewhere in your model or training procedure without causing conflicts 
                #  or retaining unnecessary parts of the graph in this function.
                
                #  Using detach() makes the operation treat self.view_weights as a constant rather than
                #  a variable needing gradient updates in this context. This prevents errors related to
                #  trying to backpropagate through parts of the graph
                #  that have already completed their updates and had their intermediate states discarded.
                loss = (render_update_mask * self.view_weights.detach() * (rgb_render - rgb_output.detach()).pow(2)).mean()
                #Using self.view_weights in the loss requires retrain_graph = True in loss.backward()
                #loss = ( render_update_mask * (rgb_render - rgb_output.detach()).pow(2)).mean()
                
                # # Debugging: Check the grad_fn attribute
                # print(f"Iteration {iter}: loss.grad_fn: {loss.grad_fn}")
                
                #loss.backward(retain_graph=True) # JA: Compute the gradient vector of the loss with respect to the trainable parameters of
                # the network, that is, the pixel value of the texture atlas
                try:
                    loss.backward()
                    #print("Backward pass completed")
                                     
                    #MJ: Computational Graph Dynamics: Each time you compute the loss and call loss.backward(),
                    # a new computational graph is created just for that computation. This graph is specific 
                    # to the operations that compute the loss for that particular iteration, 
                    # involving the current parameters and the data processed in that loop. 
                    # Once .backward() is executed, this graph is discarded (unless retain_graph=True),
                    # but the parameters themselves retain the updates applied by the optimizer.step() method.
                    
                    #MJ: Computational Graph and rgb_render: When loss.backward() is called,
                    # the gradients are computed by backpropagating through the computational graph 
                    # that was used to compute the loss. This graph includes all operations that
                    # produced rgb_render and any tensors derived from it that were used to calculate the loss.

                    # Memory Management: After the backward pass (loss.backward()), PyTorch typically frees up
                    # the memory associated with the intermediate tensors in the computational graph to
                    # save space, since these tensors (and their gradients) are no longer needed.
                    # This includes the memory for any operations and intermediate results leading up 
                    # to rgb_render. However, rgb_render itself, as a tensor resulting from your model's
                    # operations (self.mesh_model.render()), is not automatically "discarded" 
                    # but remains in memory until it goes out of scope or is overwritten.
                    
                    #Summary: So in your training loop, rgb_render is maintained after loss.backward() 
                    # unless you explicitly overwrite it or it goes out of scope. 
                    # However, the computational graph that helped compute it is cleared to free up memory,
                    # unless you specify otherwise with retain_graph=True.
                    
                    #Important Notes:   How loss.backward() Works
                    # When loss.backward() is called, it computes the gradients for all tensors in the computational graph that 
                    # have requires_grad=True, regardless of whether they are currently linked to an optimizer or not. 
                    # This is because the backward pass is solely responsible for computing gradients 
                    # based on the structure of the computational graph and the operation that produced the loss.
                    
                    #Role of the Optimizer:
                    # The optimizer, such as Adam in your case, is responsible for updating the parameters it knows about
                    # (those passed during its initialization) based on the gradients computed during the backward pass. 
                    # If a parameter is not included in the optimizer’s parameter list, it won't be updated by the optimizer,
                    # even though its gradient might still be computed if it’s part of the computational graph leading to the loss
                    # Your Scenario:
                    # Using self.view_weights: If self.view_weights has requires_grad=True and is part of the computation leading
                    # to loss, then loss.backward() will attempt to compute its gradients. 
                    # This is true even if self.view_weights is not listed in the optimizer's parameters.# 
                    
                    # Detaching self.view_weights: When you call self.view_weights.detach(), 
                    # you effectively stop loss.backward() from computing gradients for self.view_weights.
                    # This is because detach() creates a tensor that does not require gradients 
                    # and is not part of the computational graph.
                    

                except RuntimeError as e:
                    print(f"Error during backward pass at iteration {iter}: {e}")
                    raise
                #  # Debugging: Check gradients after backward pass
                # for name, param in self.mesh_model.named_parameters():
                #     if param.grad is not None:
                #         print(f"Iteration {iter}: Gradient for {name}: {param.grad.norm()}")
                        
       
                optimizer.step()
                pbar.set_description(f"zero123plus: Fitting mesh colors -Epoch {iter}, Loss: {loss.item():.7f}")

               
        #MJ:  with tqdm(range(200), desc='project_back_only_texture_atla: fitting mesh colors') as pbar
        
        return rgb_render
    #MJ:  self.project_back_max_z_normals(
        #     background=None, object_mask=self.mask, z_normals=self.normals_image[:,2:3,:,:]
        #     face_normals = self.face_normals, face_idx=self.face_idx           
        # )
    
    def project_back_max_z_normals(self):
        optimizer = torch.optim.Adam(self.mesh_model.get_params_max_z_normals(), lr=self.cfg.optim.lr, betas=(0.9, 0.99),
                                        eps=1e-15)

        #End  for v in range( len(self.thetas) )
        with  tqdm(range(300), desc='project_back_max_z_normals:fitting max_z_normals') as pbar:
            render_cache = None
            for iter in pbar:
                #MJ: Render the max_z_normals (self.meta_texure_img) which has been learned using the previous view z_normals
                # At the beginning of the for loop, self.meta_texture_img is set to 0
                if render_cache is None:
                    meta_output = self.mesh_model.render(theta=self.thetas, phi=self.phis, radius=self.radii,
                                                    background=torch.Tensor([0, 0, 0]).to(self.device),
                                                            use_meta_texture=True, render_cache=None)
                    render_cache = meta_output["render_cache"]
                else:
                    meta_output = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device), use_meta_texture=True, render_cache=render_cache)

                max_z_normals_projected = meta_output['image'][:,0:1,:,:]
                #MJ: meta_output['image'] is the projected meta_texture_img; The first channel refers to the max_z_normals in the current view
                z_normals = meta_output['normals'][:,2:3,:,:]   #MJ: Get the z component of the face normal in the current view
                #MJ: z_normals is the z component of the normal vectors of the faces seen by each view
                z_normals_mask = meta_output['mask']   #MJ: shape = (1,1,1200,1200)
                #MJ: Try blurring the object-mask "curr_z_mask" with Gaussian blurring:
                # The following code is a simply  cut and paste from project-back:
                object_mask = z_normals_mask
                # #MJ: erode the boundary of the mask
                # object_mask_v = torch.from_numpy( cv2.erode(object_mask_v[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8)) ).to(
                #                      object_mask_v.device).unsqueeze(0).unsqueeze(0)
                # # object_mask = torch.from_numpy(
                # #     cv2.erode(object_mask[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))).to(
                # #     object_mask.device).unsqueeze(0).unsqueeze(0)
                # # render_update_mask = object_mask.clone()
                render_update_mask =  object_mask.clone()
                # #MJ: dilate the bounary of the mask
                # blurred_render_update_mask_v = torch.from_numpy(
                #      cv2.dilate(render_update_mask_v[0, 0].detach().cpu().numpy(), np.ones((25, 25), np.uint8))).to(
                #      render_update_mask_v.device).unsqueeze(0).unsqueeze(0)
                # blurred_render_update_mask_v = utils.gaussian_blur(blurred_render_update_mask_v, 21, 16)
                # # Do not get out of the object
                # blurred_render_update_mask_v[object_mask_v == 0] = 0
                max_z_normals_projected  = max_z_normals_projected.clone()  *    render_update_mask.float()
                z_normals = z_normals.clone() *  render_update_mask.float()
                delta =  max_z_normals_projected -   z_normals
            # Compute the ReLU of the negative differences
                loss_v = F.relu(-delta)  # Shape: (B, 1, h, w)
                # Sum the loss over all pixels and add to total loss

                total_loss = loss_v.sum()

                optimizer.zero_grad()
                total_loss.backward() # JA: Compute the gradient vector of the loss with respect 
                                # to the trainable parameters of the network, that is, the pixel value of the
                                # texture atlas
                optimizer.step()

                pbar.set_description(f"project_max_z_normals: Fitting z_normals -Epoch {iter}, Loss: {total_loss.item():.7f}")
        #End for _ in tqdm(range(300), desc='fitting max_z_normals')
                
         
    def compute_view_weights(self, z_normals, max_z_normals, alpha=-10.0 ):        
        
        """
        Compute view weights where the weight increases exponentially as z_normals approach max_z_normals.
        
        Args:
            z_normals (torch.Tensor): The tensor containing the z_normals data.
            max_z_normals (torch.Tensor): The tensor containing the max_z_normals data.
            alpha (float): A scaling parameter that controls how sharply the weight increases (should be negative).
        
        Returns:
            torch.Tensor: The computed weights with the same shape as the input tensors (B, 1, H, W).
        """
        # Ensure inputs have the same shape
        assert z_normals.shape == max_z_normals.shape, "Input tensors must have the same shape"
        
        # Compute the difference between max_z_normals and z_normals

        delta = max_z_normals - z_normals
        # for i in range( delta.shape[0]):
        #     print(f'min delta for view-{i}:{delta[i].min()}')
        #     print(f'max  delta for view-{i}:{delta[i].max()}')
        #MJ: delta is supposed to be greater than 0; But sometimes, z_normals is greater than max_z_normals.
        # It means that project_back_max_z_normals() was not fully successful.
        
        max_z_normals = torch.where( delta >=0, max_z_normals, z_normals)
        delta_new = max_z_normals - z_normals
        # Calculate the weights using an exponential function, multiplying by negative alpha
        weights = torch.exp(alpha * delta_new)  #MJ: the max value of torch.exp(alpha * delta)   will be torch.exp(alpha * 0) = 1 
        #debug: for i in range( weights.shape[0]):
        #     print(f'min weights  for view-{i}:{weights[i].min()}')
        #     print(f'max  weights for view-{i}:{weights[i].max()}')
        # Normalize to have the desired shape (B, 1, H, W)
        #weights = weights.view(weights.size(0), 1, weights.size(1), weights.size(2))
        
        return weights
       
    
    def log_train_image(self, tensor: torch.Tensor, name: str, colormap=False):
        if self.cfg.log.log_images:
            if colormap:
                tensor = cm.seismic(tensor.detach().cpu().numpy())[:, :, :3]
            else:
                tensor = einops.rearrange(tensor, '(1) c h w -> h w c').detach().cpu().numpy()
            
            if np.any(np.isnan(tensor)) or np.any(np.isinf(tensor)):
    #     # Raise an exception if there are any NaNs or infinite values
    #      tensor = einops.rearrange(tensor, '(1) c h w -> h w c').detach().cpu().numpy()
    #      Image.fromarray( (tensor * 255).astype(np.uint8) ).save('experiments'/f'debug:NanOrInf.jpg')

                raise ValueError("Tensor contains NaNs or infinite values")
            
            Image.fromarray((tensor * 255).astype(np.uint8)).save(
                self.train_renders_path / f'debug:{name}.jpg')

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
