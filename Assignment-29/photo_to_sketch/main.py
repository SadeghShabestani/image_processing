import cv2

# Read the image in grayscale mode; 0 implies grayscale.
image = cv2.imread("photo_to_sketch/input/bill.jpeg", 0)

# Invert the image to get the negative
inverted = 255 - image

# Apply Gaussian Blur to the inverted image
blurred = cv2.GaussianBlur(inverted, (21, 21), 0)

# Invert the blurred image
inverted_blurred = 255 - blurred

# Divide the grayscale image by the inverted blurred image to create the pencil sketch effect
sketch = image / inverted_blurred

# Scale the values back up to the 0-255 range and convert to a proper format
sketch = sketch * 255

# Save the resulting sketch
cv2.imwrite("photo_to_sketch/output/bill.jpeg", sketch)
