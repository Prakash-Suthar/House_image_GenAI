import io
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from diffusers import StableDiffusionInpaintPipeline

app = FastAPI()

device = "cuda:0"
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)
pipe = pipe.to(device)


class InpaintRequest(BaseModel):
    mask_points: str
    prompt: str


def generate_binary_mask(mask_points, image_width=1023, image_height=592):
    mask_coords = [float(coord) for coord in mask_points.split(",")]
    shape = Image.new('L', (image_width, image_height), 0)
    draw = ImageDraw.Draw(shape)
    draw.polygon(mask_coords, outline=255, fill=255)
    mask = np.array(shape)
    return mask


def inpaint(image: Image.Image, mask_points: str, prompt: str):
    image = image.resize((512, 512))
    mask_image = generate_binary_mask(mask_points)
    mask_image = Image.fromarray(mask_image)
    mask_image = mask_image.resize((512, 512))
    output = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
    return Image.fromarray(output)


@app.post("/inpaint")
async def inpaint_route(
    image: UploadFile = File(...),
    mask_points: str = Form(...),
    prompt: str = Form(...),
):
    contents = await image.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    result = inpaint(image, mask_points, prompt)
    return {"result": result}
