import os
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch import autocast

from transformers import CLIPTextModel, CLIPTokenizer
from transformers import DPTForDepthEstimation, DPTFeatureExtractor

from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers.scheduling_pndm import PNDMScheduler

from diffusion import DiffusionPipeline

class Depth2ImgPipeline(DiffusionPipeline):
    def __init__(self,
                 vae,
                 tokenizer,
                 text_encoder,
                 unet,
                 scheduler,
                 depth_feature_extractor,
                 depth_estimator):

        super().__init__(vae, tokenizer, text_encoder, unet, scheduler)

        self.depth_feature_extractor = depth_feature_extractor
        self.depth_estimator = depth_estimator


    def get_depth_mask(self, img):
        if not isinstance(img, list):
            img = [img]

        width, height = img[0].size

        # pre-process the input image and get its pixel values
        pixel_values = self.depth_feature_extractor(img, return_tensors="pt").pixel_values

        # use autocast for automatic mixed precision (AMP) inference
        with autocast('cpu'):
            depth_mask = self.depth_estimator(pixel_values).predicted_depth

        # get the depth mask
        depth_mask = torch.nn.functional.interpolate(depth_mask.unsqueeze(1),
                                                     size=(height//8, width//8),
                                                     mode='bicubic',
                                                     align_corners=False)

        # scale the mask to range [-1, 1]
        depth_min = torch.amin(depth_mask, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_mask, dim=[1, 2, 3], keepdim=True)
        depth_mask = 2.0 * (depth_mask - depth_min) / (depth_max - depth_min) - 1.0
        depth_mask = depth_mask.to(self.device)

        # replicate the mask for classifier free guidance
        depth_mask = torch.cat([depth_mask] * 2)
        return depth_mask




    def denoise_latents(self,
                        img,
                        prompt_embeds,
                        depth_mask,
                        strength,
                        num_inference_steps=50,
                        guidance_scale=7.5,
                        height=512, width=512):

        # clip the value of strength to ensure strength lies in [0, 1]
        strength = max(min(strength, 1), 0)

        # compute timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        init_timestep = int(num_inference_steps * strength)
        t_start = num_inference_steps - init_timestep

        timesteps = self.scheduler.timesteps[t_start: ]
        num_inference_steps = num_inference_steps - t_start

        latent_timestep = timesteps[:1].repeat(1)

        latents = self.encode_img_latents(img, latent_timestep)

        # use autocast for automatic mixed precision (AMP) inference
        with autocast('cpu'):
            for i, t in tqdm(enumerate(timesteps)):
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = torch.cat([latent_model_input, depth_mask], dim=1)

                # predict noise residuals
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds)['sample']

                # separate predictions for unconditional and conditional outputs
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

                # perform guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # remove the noise from the current sample i.e. go from x_t to x_{t-1}
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents


    def __call__(self,
                 prompt,
                 img,
                 strength=0.8,
                 num_inference_steps=50,
                 guidance_scale=7.5,
                 height=512, width=512):


        prompt_embeds = self.get_prompt_embeds(prompt)

        depth_mask = self.get_depth_mask(img)

        latents = self.denoise_latents(img,
                                       prompt_embeds,
                                       depth_mask,
                                       strength,
                                       num_inference_steps,
                                       guidance_scale,
                                       height, width)

        img = self.decode_img_latents(latents)

        img = self.transform_img(img)

        return img
