# import cv2
# import matplotlib.pyplot as plt
# import gradio as gr

# # Function to inpaint an image with a provided mask
# def inpaint_image(input_image, mask_image):
#     # Convert input images to OpenCV format
#     damaged_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
#     mask = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

#     # Inpainting using the TELEA algorithm
#     output1 = cv2.inpaint(damaged_image, mask, 1, cv2.INPAINT_TELEA)

#     # Inpainting using the NS algorithm
#     output2 = cv2.inpaint(damaged_image, mask, 1, cv2.INPAINT_NS)

#     # Prepare the inpainted images for display
#     inpainted_telea = cv2.cvtColor(output1, cv2.COLOR_RGB2BGR)
#     inpainted_ns = cv2.cvtColor(output2, cv2.COLOR_RGB2BGR)

#     return inpainted_telea, inpainted_ns

# # Gradio interface
# iface = gr.Interface(
#     fn=inpaint_image,
#     inputs=[
#         gr.Image(type="pil", tool = 'sketch', label="Damaged Image"),
#         # gr.Image(type="pil", label="Mask (Edit with Brush)"),
#     ],
#     outputs=[
#         gr.Image(type="pil", label="Inpainting (TELEA)"),
#         gr.Image(type="pil", label="Inpainting (NS)"),
#     ],
#     live=True,
#     title="Image Inpainting with Brush Editable Mask",
#     description="Upload a damaged image and create a mask using the brush edit option on the input image. The model will inpaint the image using both TELEA and NS algorithms.",
# )

# iface.launch()

import cv2
import matplotlib.pyplot as plt
import gradio as gr

# Function to inpaint an image with a provided mask
def inpaint_image(input_image, mask_image):
    # Convert input images to OpenCV format
    damaged_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

    # Inpainting using the TELEA algorithm
    output1 = cv2.inpaint(damaged_image, mask, 1, cv2.INPAINT_TELEA)

    # Inpainting using the NS algorithm
    output2 = cv2.inpaint(damaged_image, mask, 1, cv2.INPAINT_NS)

    # Prepare the inpainted images for display
    inpainted_telea = cv2.cvtColor(output1, cv2.COLOR_RGB2BGR)
    inpainted_ns = cv2.cvtColor(output2, cv2.COLOR_RGB2BGR)

    return inpainted_telea, inpainted_ns

# Gradio interface
iface = gr.Interface(
    fn=inpaint_image,
    inputs=[
        gr.Image(source = 'upload',type="pil", tool="sketchpad", label="Damaged Image (Sketch & Edit)"),
    ],
    outputs=[
        gr.Image(type="pil", label="Inpainting (TELEA)"),
        gr.Image(type="pil", label="Inpainting (NS)"),
    ],
    live=True,
    title="Image Inpainting with Sketch and Editable Mask",
    description="Upload a damaged image and create a mask using the sketch and edit feature on the input image. The model will inpaint the image using both TELEA and NS algorithms.",
)

iface.launch()

