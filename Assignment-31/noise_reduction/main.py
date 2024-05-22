import cv2
import numpy as np


def noise_reduction(image, kernel_size):
    # Get the dimensions of the image
    rows, cols = image.shape

    # Create an empty result image with the same dimensions as the input image
    result = np.zeros((rows, cols), dtype=np.uint8)

    # Define the horizontal edge detection kernel
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8) / (
        kernel_size * kernel_size
    )

    # Calculate the offset for the kernel's center
    offset = (kernel_size - 1) // 2

    # Iterate over each pixel in the image (excluding the borders)
    for row in range(offset, rows - offset):
        for col in range(offset, cols - offset):
            # Extract the 3x3 neighborhood around the current pixel
            small = image[
                row - offset : row + (offset + 1), col - offset : col + (offset + 1)
            ]
            # Apply the kernel to the neighborhood
            average = np.sum(kernel * small)
            # Set the result pixel value to the computed average
            result[row, col] = average

    return result


# Load the noisy images in grayscale
noisy_skeleton = cv2.imread(
    "image_processing/Assignment-31/noise_reduction/input/noisy_skeleton.png",
    cv2.IMREAD_GRAYSCALE,
)
noisy_image = cv2.imread(
    "image_processing/Assignment-31/noise_reduction/input/noisy_image.png",
    cv2.IMREAD_GRAYSCALE,
)
noisy_board = cv2.imread(
    "image_processing/Assignment-31/noise_reduction/input/noisy_board.png",
    cv2.IMREAD_GRAYSCALE,
)

# Apply noise reduction with different kernel sizes and display the results
for img, title in [
    (noisy_skeleton, "Skeleton"),
    (noisy_image, "Image"),
    (noisy_board, "Board"),
]:
    for size in [3, 5, 15]:
        result = noise_reduction(img, size)
        cv2.imshow(f"{title} {size}x{size}", result)
        cv2.waitKey(0)
        cv2.imwrite(
            f"image_processing/Assignment-31/noise_reduction/output/{title.lower()}_{size}x{size}.png",
            result,
        )
