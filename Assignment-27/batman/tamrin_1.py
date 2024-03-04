import cv2

# Load the original image
image = cv2.imread('batman_orginal.jpg')

# Convert the image to grayscale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a binary image
_, image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)

# Get the dimensions of the image
rows, cols = image.shape

# Add text to the image
# Parameters: image, text, position, font, scale, color
cv2.putText(image, "BATMAN", (rows - 250, cols // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, 255)

# Save the modified image
cv2.imwrite("batman_orginal.jpg", image)