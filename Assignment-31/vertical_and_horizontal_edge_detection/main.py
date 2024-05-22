import cv2
import numpy as np


def edge_detection_vertical(image):
    # Get the dimensions of the image
    rows, cols = image.shape

    # Create an empty result image with the same dimensions as the input image
    result = np.zeros((rows, cols), dtype=np.uint8)

    # Define the vertical edge detection kernel
    kernel = np.array([[0.5, 0, -0.5], [0.5, 0, -0.5], [0.5, 0, -0.5]])

    # Iterate over each pixel in the image (excluding the borders)
    for row in range(1, rows - 1):
        for col in range(1, cols - 1):
            # Extract the 3x3 neighborhood around the current pixel
            small = image[row - 1 : row + 2, col - 1 : col + 2]
            # Apply the kernel to the neighborhood
            average = np.abs(np.sum(kernel * small))
            # Set the result pixel value to the computed average
            result[row, col] = average

    return result


def edge_detection_horizontal(image):
    # Get the dimensions of the image
    rows, cols = image.shape

    # Create an empty result image with the same dimensions as the input image
    result = np.zeros((rows, cols), dtype=np.uint8)

    # Define the horizontal edge detection kernel
    kernel = np.array([[-0.5, -0.5, -0.5], [0, 0, 0], [0.5, 0.5, 0.5]])

    # Iterate over each pixel in the image (excluding the borders)
    for row in range(1, rows - 1):
        for col in range(1, cols - 1):
            # Extract the 3x3 neighborhood around the current pixel
            small = image[row - 1 : row + 2, col - 1 : col + 2]
            # Apply the kernel to the neighborhood
            average = np.abs(np.sum(kernel * small))
            # Set the result pixel value to the computed average
            result[row, col] = average

    return result


# Read the image in grayscale
image = cv2.imread(
    "image_processing/Assignment-31/vertical_and_horizontal_edge_detection/input/house.jpg",
    cv2.IMREAD_GRAYSCALE,
)

# Perform vertical edge detection on the image
edge_detection_vertical_instance = edge_detection_vertical(image=image)

# Display the result
cv2.imshow("result", edge_detection_vertical_instance)
cv2.waitKey(0)
# Save the result to a file
cv2.imwrite(
    "image_processing/Assignment-31/vertical_and_horizontal_edge_detection/output/house_vertical.jpg",
    edge_detection_vertical_instance,
)


# Perform horizontal edge detection on the image
edge_detection_horizontal_instance = edge_detection_horizontal(image=image)

# Display the result
cv2.imshow("result", edge_detection_horizontal_instance)
cv2.waitKey(0)
# Save the result to a file
cv2.imwrite(
    "image_processing/Assignment-31/vertical_and_horizontal_edge_detection/output/house_horizontal.jpg",
    edge_detection_horizontal_instance,
)
