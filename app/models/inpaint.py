# # repair damaged images
# import cv2
# import matplotlib.pyplot as plt

# damaged_image_path = r"C:\Users\praka\Downloads\dog.tiff"
# damaged_image = cv2.imread(damaged_image_path)

# mask_path = r"C:\Users\praka\Downloads\dog_mask.tiff"
# mask = cv2.imread(mask_path, 0)

# damaged_image = cv2.cvtColor(damaged_image, cv2.COLOR_BGR2RGB)

# output1 = cv2.inpaint(damaged_image, mask, 1, cv2.INPAINT_TELEA)
# output2 = cv2.inpaint(damaged_image, mask, 1, cv2.INPAINT_NS)

# img = [damaged_image, mask, output1, output2]
# titles = ['damaged image', 'mask', 'TELEA', 'NS']

# for i in range(4):
#     plt.subplot(2, 2, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.title(titles[i])
#     plt.imshow(img[i])
# plt.show()


import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np

# Specify the paths to your damaged image and mask (in PNG or JPG format)
damaged_image_path = r"C:\Users\praka\Downloads\d23.jpeg"

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
    mask_pth = r"E:\codified\gen_ai\House_image_genAI\app\models\mask_img.png" 
    cv2.imwrite(mask_pth, mask)
    
    return mask_pth

mask_points = [626, 209, 626, 257, 658, 257, 658, 209]

# Read the damaged image
damaged_image = cv2.imread(damaged_image_path)
damaged_image = cv2.resize(damaged_image, (950,550))

mask_path = generate_binary_mask(mask_points)

# Read the binary mask
mask = cv2.imread(mask_path, 0)

# Convert the image to RGB format (if not already)
damaged_image = cv2.cvtColor(damaged_image, cv2.COLOR_BGR2RGB)

# Inpainting using the TELEA algorithm
output1 = cv2.inpaint(damaged_image, mask, 1, cv2.INPAINT_TELEA)

# Inpainting using the NS algorithm
output2 = cv2.inpaint(damaged_image, mask, 1, cv2.INPAINT_NS)

# Prepare the images for display
img = [damaged_image, mask, output1, output2]
titles = ['Damaged Image', 'Mask', 'Inpainting (TELEA)', 'Inpainting (NS)']

# Display the images
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.title(titles[i])
    plt.imshow(img[i])

plt.show()
