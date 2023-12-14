
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw
# from segment_anything import SamPredictor, sam_model_registry
import gradio as gr
device = "cuda"

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    revision="fp16",
    torch_dtype=torch.float16,
)

pipe = pipe.to(device)

with gr.Blocks() as demo:
    with gr.Row():
        input_img = gr.Image(label="Input")
        mask_points = gr.Textbox(lines=1, label="Mask Points")

    with gr.Row():
        prompt_text = gr.Textbox(lines=1, label="Prompt")
        output_img = gr.Image(label="Output")

    with gr.Row():
        submit = gr.Button("Submit")

    def generate_binary_mask(mask_points):
        # Convert the comma-separated string of coordinates to a list of integers
        mask_coords = [int(coord) for coord in mask_points.split(",")]
        image_width = 950
        image_height = 550

        # Create a black and white image with a black background
        shape = Image.new('L', (image_width, image_height), 0)

        # Create a white mask in the image
        draw = ImageDraw.Draw(shape)
        draw.polygon(mask_coords, outline=255, fill=255)

        # Convert the grayscale image to a binary mask
        mask = np.array(shape)

        return mask

    def inpaint(image, mask, prompt):
        image = Image.fromarray(image)
        image = image.resize((512, 512))
        mask_image = generate_binary_mask(mask)
        mask_image = Image.fromarray(mask_image)
        mask_image = mask_image.resize((512, 512))
        output = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask_image
        ).images[0]

        return output

    submit.click(
        inpaint,
        inputs=[input_img, mask_points, prompt_text],
        outputs=[output_img],
    )

if __name__ == "__main__":
    demo.launch(debug = True)
