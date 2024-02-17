# import os
# from PIL import Image

# # from utils.instance import model_use

# img = r"E:\codified\gen_ai\House_image_genAI\app\assests\input_img\master1 (1).png"
# im = Image.open(img)
# # im.show()

# # model_use("two tigers", img)[0]

# import torch
# import requests
# from PIL import Image
# from diffusers import StableDiffusionDepth2ImgPipeline

# pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-2-depth",
#     torch_dtype=torch.float16,
# ).to("cpu")

# prompt = "a house with colorfull lightning"
# t = pipe(prompt=prompt, image=im, negative_prompt=None, strength=0.7).images[0]

# t.show()



import os
from PIL import Image
import torch
from diffusers import StableDiffusionDepth2ImgPipeline

# Load the image
img_path = r"E:\codified\gen_ai\House_image_genAI\app\assests\input_img\master1 (1).png"
im = Image.open(img_path)

# Create the StableDiffusionDepth2ImgPipeline with float32 precision
pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-depth",
    torch_dtype=torch.float32,  # Set the data type to float32
).to("cpu")

# Define your prompt
prompt = "a house with colorful lighting"

# Generate the image with the specified prompt
generated_image = pipe(prompt=prompt, image=im, negative_prompt=None, strength=0.7).images[0]

# Display the generated image
generated_image.show()
