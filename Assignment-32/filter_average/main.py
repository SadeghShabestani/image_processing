import cv2
import numpy as np


# Function to apply different averaging filters to an image
def apply_filters(image, kernel_size):
    # Define averaging kernels with different normalization factors
    kernel_normal = np.ones((kernel_size, kernel_size), dtype=np.uint8) / 5
    kernel_high = np.ones((kernel_size, kernel_size), dtype=np.uint8) / 1
    kernel_low = np.ones((kernel_size, kernel_size), dtype=np.uint8) / 0.04

    # Apply the kernels to the image
    result_normal = cv2.filter2D(image, -1, kernel_normal)
    result_high = cv2.filter2D(image, -1, kernel_high)
    result_low = cv2.filter2D(image, -1, kernel_low)

    # Concatenate the original and filtered images for comparison
    result_list = np.hstack((image, result_normal, result_high, result_low))
    # Save the concatenated result image
    cv2.imwrite(
        f"image_processing/Assignment-32/filter_average/output/filter_{kernel_size}x{kernel_size}.tif",
        result_list,
    )

    return result_list


# Read the input image
image = cv2.imread("image_processing/Assignment-32/filter_average/input/1.tif")

# Apply filters with a 3x3 kernel and display the result
filter_instance_3x3 = apply_filters(image, 3)
cv2.imshow("result", filter_instance_3x3)
cv2.waitKey(0)

# Apply filters with a 5x5 kernel and display the result
filter_instance_5x5 = apply_filters(image, 5)
cv2.imshow("result", filter_instance_5x5)
cv2.waitKey(0)
