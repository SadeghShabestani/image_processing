import cv2
import numpy as np

# Define the dimensions of the gradient image
width = 800
height = 600

# Create an empty image
gradient_image = np.zeros((height, width), dtype=np.uint8)

# Generate the gradient by varying intensity linearly across columns
for x in range(width):
    gradient_image[:, x] = np.linspace(255, 0, height, dtype=np.uint8)

# Display the gradient image
cv2.imwrite('Gradient.jpg', gradient_image)
cv2.waitKey()