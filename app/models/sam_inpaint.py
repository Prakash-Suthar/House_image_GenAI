import gradio as gr
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw
from segment_anything import SamPredictor, sam_model_registry

device = "cuda"
# sam_checkpoint = r"C:\Users\praka\Downloads\sam_vit_h_4b8939.pth"
# model_type = "vit_h"
# sam = sam_model_registry [model_type](checkpoint= sam_checkpoint)
# sam.to(device)
# predictor = SamPredictor(sam)
selected_pixel = []

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting", 
    torch_dtype=torch.float16, 
)

pipe =  pipe.to(device)

with gr.Blocks() as demo: 
    with gr.Row():
        input_img = gr.Image(label="Input")
        mask_img = gr.Image(label="Mas")
        output_img = gr.Image(label="Output")

    with gr.Blocks():
            prompt_text = gr. Textbox(lines=1, label="Prompt")
            mask_points = gr.Textbox(lines =1, label="segmentation coords")
    with gr.Row():

        submit = gr.Button("Submit")

    
    def generate_binary_mask(mask_points):
        image_width = 950
        image_height = 550
        # Create a black and white image with a black background
        shape = Image.new('L', (image_width, image_height), 0)

        # Create a white mask in the image
        draw = ImageDraw.Draw(shape)
        draw.polygon(mask_points, outline=255, fill=255)

        # Convert the grayscale image to a binary mask
        mask = np.array(shape)

        return mask
        
    def inpaint (image, mask, prompt):
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)
        image = image.resize((512, 512))
        mask = mask.resize((512, 512))
        output = pipe(
            prompt=prompt, 
            image=image, 
            mask_image=mask
            ).images[0]
        
        return output
    
    input_img.select(generate_binary_mask, [input_img], [mask_points], [mask_img])
    
    submit.click(
        inpaint, 
        inputs = [input_img, mask_img, prompt_text, mask_points],
        outputs=[output_img],
    )
    
    
if __name__ == "__main__":
    demo.launch()


# import cv2
# import matplotlib.pyplot as plt
# from PIL import Image, ImageDraw
# import numpy as np



# def generate_binary_mask(image_width, image_height, mask_points):
#     # Create a black and white image with a black background
#     shape = Image.new('L', (image_width, image_height), 0)

#     # Create a white mask in the image
#     draw = ImageDraw.Draw(shape)
#     draw.polygon(mask_points, outline=255, fill=255)

#     # Convert the grayscale image to a binary mask
#     binary_mask = np.array(shape)

#     return binary_mask

# # Example usage
# image_width = 950  # Replace with your image width
# image_height = 550  # Replace with your image height
# mask_points = [162, 327, 161, 328, 159, 328, 158, 329, 151, 329, 150, 330, 130, 330, 129, 331, 77, 331, 76, 332, 71, 332, 71, 423, 275, 423, 276, 422, 276, 420, 277, 419, 277, 415, 278, 414, 278, 406, 279, 405, 279, 403, 280, 402, 280, 400, 281, 399, 281, 398, 282, 397, 282, 395, 283, 394, 283, 335, 282, 334, 282, 333, 278, 329, 277, 329, 276, 328, 272, 328, 271, 327]  # Replace with your mask points

# single_segment_mask = generate_binary_mask(image_width, image_height, mask_points)

# plt.imshow(single_segment_mask)
# plt.axis('off')
# plt.show()