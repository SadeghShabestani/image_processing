import cv2

# Load the image
image = cv2.imread('Assignment-26/human.jpg')

# Check if the image has been loaded successfully
if image is None:
    print('Error: Unable to load the image.')
else:
    # Rotate the image 180 degrees
    rotated_image = cv2.rotate(image, cv2.ROTATE_180)

    # Display the original and inverted images
    cv2.imwrite('inverted_image_human.jpg',rotated_image)
    cv2.waitKey()