import cv2
import numpy as np
import glob
import os


# Function to apply histogram equalization to an image
def equalizer_histogram(image, name):
    # Apply histogram equalization
    equalized = cv2.equalizeHist(image)

    # Concatenate the original image and the equalized image side by side
    result = np.hstack((image, equalized))

    # Save the result image with the specified name
    cv2.imwrite(
        f"image_processing/Assignment-32/histogram_equalization/output/equalizer_{name}.png",
        result,
    )
    return result


# Function to apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image
def clahe_create(image, name):
    # Create CLAHE object with specified clip limit and tile grid size
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Apply CLAHE to the image
    clahe_applied = clahe.apply(image)

    # Concatenate the original image and the CLAHE result side by side
    result = np.hstack((image, clahe_applied))

    # Save the result image with the specified name
    cv2.imwrite(
        f"image_processing/Assignment-32/histogram_equalization/output/clahe_{name}.png",
        result,
    )
    return result


# Function to read images from a specified directory
def read_images_from_directory(directory_path):
    # Get a list of all image file paths in the directory
    image_paths = glob.glob(os.path.join(directory_path, "*.png"))

    images = []
    for path in image_paths:
        # Read each image in grayscale mode
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            # Extract the image name without the directory path
            image_name = os.path.basename(path).split(".")[0]
            images.append((image, image_name))
    return images


# Directory containing input images
input_directory = "image_processing/Assignment-32/histogram_equalization/input/"

# Read images from the input directory
images = read_images_from_directory(input_directory)

# Apply histogram equalization and CLAHE to each image and display the result
for image, name in images:
    # Apply histogram equalization
    equalized_image = equalizer_histogram(image, name)
    cv2.imshow("result", equalized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Apply CLAHE
    clahe_image = clahe_create(image, name)
    cv2.imshow("result", clahe_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
