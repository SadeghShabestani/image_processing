import cv2
import numpy as np


# Edge detection filter
def edge_detection(image):
    # Define the edge detection kernel
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    # Apply the kernel to the image using filter2D
    detection = cv2.filter2D(image, -1, kernel)
    # Concatenate the edge detection result with the original image for comparison
    result = np.hstack((detection, image))
    # Save the result image
    cv2.imwrite(
        "image_processing/Assignment-32/convolution_2D/output/tiger_edge_detection.jpg",
        result,
    )
    return result


# Sharpening filter
def sharpening_filter(image):
    # Define the sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # Apply the kernel to the image using filter2D
    detection = cv2.filter2D(image, -1, kernel)
    # Concatenate the sharpening result with the original image for comparison
    result = np.hstack((detection, image))
    # Save the result image
    cv2.imwrite(
        "image_processing/Assignment-32/convolution_2D/output/tiger_sharpening_filter.jpg",
        result,
    )
    return result


# Emboss filter
def emboss_filter(image):
    # Define the embossing kernel
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    # Apply the kernel to the image using filter2D
    detection = cv2.filter2D(image, -1, kernel)
    # Concatenate the embossing result with the original image for comparison
    result = np.hstack((detection, image))
    # Save the result image
    cv2.imwrite(
        "image_processing/Assignment-32/convolution_2D/output/tiger_emboss_filter.jpg",
        result,
    )
    return result


# Identity filter
def identity_filter(image):
    # Define the identity kernel (which doesn't change the image)
    kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    # Apply the kernel to the image using filter2D
    detection = cv2.filter2D(image, -1, kernel)
    # Concatenate the identity result with the original image for comparison
    result = np.hstack((detection, image))
    # Save the result image
    cv2.imwrite(
        "image_processing/Assignment-32/convolution_2D/output/tiger_identity_filter.jpg",
        result,
    )
    return result


# Gaussian-like blur filter
def gaussian_like_blur_filter(image):
    # Define the Gaussian-like blur kernel
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)
    # Normalize the kernel
    kernel /= np.sum(kernel)
    # Apply the kernel to the image using filter2D
    detection = cv2.filter2D(image, -1, kernel)
    # Concatenate the blur result with the original image for comparison
    result = np.hstack((detection, image))
    # Save the result image
    cv2.imwrite(
        "image_processing/Assignment-32/convolution_2D/output/tiger_gaussian_like_blur_filter.jpg",
        result,
    )
    return result


# Read the input image
image = cv2.imread("image_processing/Assignment-32/convolution_2D/input/tiger.jpeg")

# Apply the edge detection filter and display the result
edge_detection_instance = edge_detection(image=image)
cv2.imshow("result", edge_detection_instance)
cv2.waitKey(0)

# Apply the sharpening filter and display the result
sharpening_filter_instance = sharpening_filter(image=image)
cv2.imshow("result", sharpening_filter_instance)
cv2.waitKey(0)

# Apply the emboss filter and display the result
emboss_filter_instance = emboss_filter(image=image)
cv2.imshow("result", emboss_filter_instance)
cv2.waitKey(0)

# Apply the identity filter and display the result
identity_filter_instance = identity_filter(image=image)
cv2.imshow("result", identity_filter_instance)
cv2.waitKey(0)

# Apply the Gaussian-like blur filter and display the result
gaussian_like_blur_filter_instance = gaussian_like_blur_filter(image=image)
cv2.imshow("result", gaussian_like_blur_filter_instance)
cv2.waitKey(0)
