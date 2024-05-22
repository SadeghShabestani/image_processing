import cv2
import numpy as np


def edge_detection(image):
    # Get the dimensions of the image
    rows, cols = image.shape

    # Create an empty result image with the same dimensions as the input image
    result = np.zeros((rows, cols), dtype=np.uint8)

    # Define the edge detection kernel (Laplacian kernel)
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

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
    "image_processing/Assignment-31/edge_detection/input/lion.png",
    cv2.IMREAD_GRAYSCALE,
)

# Perform edge detection on the image
edge_detection_instance = edge_detection(image=image)

# Display the result
cv2.imshow("result", edge_detection_instance)
cv2.waitKey(0)
# Save the result to a file
cv2.imwrite(
    "image_processing/Assignment-31/edge_detection/output/lion_result.png",
    edge_detection_instance,
)
