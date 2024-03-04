import cv2
import numpy as np
import imageio

# Load the original image
image = cv2.imread('tv_orginal.jpg')

# Convert the image to grayscale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Get the dimensions of the image
rows, cols = image.shape

# Create a list to store frames for GIF
gif_frames = []

while True:

    # Generate noise with the same shape as the region to be replaced
    nois = np.random.random((275, 345)) * 255
    nois = np.array(nois, dtype=np.uint8)

    # Replace the region in the image with the generated noise
    image[75:350, 80:425] = nois

    # Convert the grayscale image to BGR
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Append the current frame to the list
    gif_frames.append(color_image)

    # Display the modified image
    cv2.imshow("output", color_image)

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


# Save the list of frames as a GIF
imageio.mimsave("tv_noise.gif", gif_frames)

# Close all windows
cv2.destroyAllWindows()