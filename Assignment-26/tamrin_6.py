import cv2
import numpy as np

# Load the image
image = cv2.imread('Assignment-26/bil.jpg')

# Check if the image is loaded successfully
if image is None:
    print('Error: Unable to load image.')
else:
    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    treshold = 120
    height, width = image.shape

    for i in range(250):
        if i <= 100:
            for j in range(100-i, 250-i):
                if j >= 0:
                    image[i, j] = 0
        else:
            for j in range(0, 250-i):
                if j >= 0:
                    image[i, j] = 0

    image[560:60, 0:5] = 0
    image[0:60, 501:5] = 0
    image[0:4, 0:5] = 0

    # Save and display the modified image
    cv2.imwrite('bill.jpg', image)
    cv2.waitKey()