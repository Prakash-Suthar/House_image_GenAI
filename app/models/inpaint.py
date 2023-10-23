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

# Specify the paths to your damaged image and mask (in PNG or JPG format)
damaged_image_path = r"C:\Users\praka\Downloads\dog.png"
mask_path = r"C:\Users\praka\Downloads\dog_mask.png"

# Read the damaged image and mask
damaged_image = cv2.imread(damaged_image_path)
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
