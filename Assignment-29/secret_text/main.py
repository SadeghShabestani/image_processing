import cv2

# Read the first image from the specified path
image_1 = cv2.imread("secret_text/input/image_1.png")
image_2 = cv2.imread("secret_text/input/image_2.png")

# Subtract the first image from the second image. 
result = cv2.subtract(image_2, image_1)

# Invert the result to make the subtracted area white. This can make any hidden text
result = 255 - result

# Write the result to a file in the output directory.
cv2.imwrite("secret_text/output/result.png", result)