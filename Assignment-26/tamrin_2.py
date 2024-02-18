import cv2

# Load the image
image = cv2.imread('Assignment-26/person.jpg')

# Check if the image has been loaded successfully
if image is None:
    print('Error: Unable to load the image.')
else:
    inverted_image = 255 - image

    # Display the original and inverted images
    cv2.imwrite('inverted_image_person.jpg',inverted_image)
    cv2.waitKey()