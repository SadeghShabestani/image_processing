import cv2
import numpy as np

# Read two images
image_1 = cv2.imread("face_morphing/input/img1.jpg")
image_2 = cv2.imread("face_morphing/input/img2.jpg")

# Resize the first image to match the dimensions of the second image
image_1 = cv2.resize(image_1, (image_2.shape[1], image_2.shape[0]))

# Convert the images to float32 for proper blending
image_1 = image_1.astype(np.float32)
image_2 = image_2.astype(np.float32)

# More weight to the first image
morph_25_percent_img1 = (image_1 * 0.25) + (image_2 * 0.75)
# Equal weight to both images
morph_50_percent = (image_1 * 0.5) + (image_2 * 0.5)
# More weight to the second image
morph_75_percent_img1 = (image_1 * 0.75) + (image_2 * 0.25)

# Convert the blended images back to uint8
morph_25_percent_img1 = morph_25_percent_img1.astype(np.uint8)
morph_50_percent = morph_50_percent.astype(np.uint8)
morph_75_percent_img1 = morph_75_percent_img1.astype(np.uint8)

# Concatenate the original and morphed images side by side
combined_image_sequence = np.concatenate(
    (image_1, morph_25_percent_img1, morph_50_percent, morph_75_percent_img1, image_2),
    axis=1
)

# Optionally, save the concatenated image sequence to a file
cv2.imwrite("face_morphing/output/combined_image.jpeg", combined_image_sequence)