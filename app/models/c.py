import inspect
from typing import List, Optional, Union
from PIL import Image, ImageDraw
import numpy as np
import torch
import PIL 
import gradio as gr
from diffusers import StableDiffusionInpaintPipeline
from transformers import PreTrainedModel

device = "cuda:0"
print(device)


model_path = "runwayml/stable-diffusion-inpainting"

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_path,
    variant="fp16",
    torch_dtype=torch.float16,
).to(device)

import requests
from io import BytesIO

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = PIL.Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

img_url = r"C:\Users\praka\Downloads\dog.png"
mask_url = r"C:\Users\praka\Downloads\dog_mask.png"
img_url = Image.open(img_url)
mask_url = Image.open(mask_url)
image = img_url.resize((512, 512))
# image.show()


mask_image = mask_url.resize((512, 512))
# mask_image.show()

prompt = "a mecha robot sitting on a bench"

guidance_scale=7.5
num_samples = 3
generator = torch.Generator(device=device).manual_seed(0) # change the seed to get different results

images = pipe(
    prompt=prompt,
    image=image,
    mask_image=mask_image,
    guidance_scale=guidance_scale,
    generator=generator,
    num_images_per_prompt=num_samples,
).images

# insert initial image in the list so we can compare side by side
images.insert(0, image)

image_grid(images, 1, num_samples + 1)

"""### Gradio Demo"""

def predict(dict, prompt):
  image = dict['image'].convert("RGB").resize((512, 512)).float()
  mask_image = dict['mask'].convert("RGB").resize((512, 512)).float()
  images = pipe(prompt=prompt, image=image, mask_image=mask_image).images
  return(images[0])

gr.Interface(
    predict,
    title = 'Stable Diffusion In-Painting',
    inputs=[
        gr.Image(source = 'upload', tool = 'sketch', type = 'pil'),
        gr.Textbox(label = 'prompt')
    ],
    outputs = [
        gr.Image()
        ]
).launch()

