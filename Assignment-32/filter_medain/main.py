import cv2
import numpy as np
import glob
import os


# Function to apply median blur filter to an image
def apply_filters(image, name):
    # Apply a median blur with a kernel size of 3
    blur = cv2.medianBlur(image, 3)

    # Concatenate the original image and the blurred image side by side
    result = np.hstack((image, blur))

    # Save the result image with the specified name
    cv2.imwrite(
        f"image_processing/Assignment-32/filter_medain/output/filter_{name}.png",
        result,
    )
    return result


# Function to read images from a specified directory
def read_images_from_directory(directory_path):
    # Get a list of all image file paths in the directory
    image_paths = glob.glob(os.path.join(directory_path, "*.png"))

    images = []
    for path in image_paths:
        # Read each image and store it in the list
        image = cv2.imread(path)
        if image is not None:
            # Extract the image name without the directory path
            image_name = os.path.basename(path).split(".")[0]
            images.append((image, image_name))
    return images


# Directory containing input images
input_directory = "image_processing/Assignment-32/filter_medain/input/"

# Read images from the input directory
images = read_images_from_directory(input_directory)

# Apply the median blur filter to each image and display the result
for image, name in images:
    filtered_image = apply_filters(image, name)
    cv2.imshow("result", filtered_image)
    cv2.waitKey(0)
