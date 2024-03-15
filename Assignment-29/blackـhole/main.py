import cv2
import numpy as np
import os

# Create paths for each directory containing images
directories = [f"blackـhole/input/{i}" for i in range(1, 5)]
images = []

# Process images for each directory
for directory_index, directory in enumerate(directories):
    image_paths = os.listdir(directory)
    sum_of_images = None
    
    # Read each image, convert to grayscale, and add to sum
    for img_name in image_paths:
        img_path = os.path.join(directory, img_name)
        image = cv2.imread(img_path)
        image = image.astype(np.float32)
        
        # Initialize sum_of_images with the first image
        if sum_of_images is None:
            sum_of_images = np.zeros(image.shape, np.float32)
        
        # Add the current image to the sum
        sum_of_images += image

    # Calculate the average image for the current directory
    average_image = sum_of_images / len(image_paths)
    average_image_uint8 = average_image.astype(np.uint8)
    
    # Save the average image
    cv2.imwrite(f"blackـhole/output/result{directory_index + 1}.jpg", average_image_uint8)

# Read the average images from each directory
results = [cv2.imread(f"blackـhole/output/result{i}.jpg") for i in range(1, 5)]

# Concatenate the images horizontally in pairs
output1 = np.concatenate((results[0], results[1]), axis=1)
output2 = np.concatenate((results[2], results[3]), axis=1)

# Concatenate the resulting pairs vertically to get the final output
final_output = np.concatenate((output1, output2), axis=0)

# Save the final output
cv2.imwrite("blackـhole/output/final_result.jpeg", final_output)