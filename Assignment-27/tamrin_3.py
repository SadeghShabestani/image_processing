import cv2
import numpy as np
import imageio

# Load the original image
image = cv2.imread("view_orginal.png")

# Get the dimensions of the image
rows, cols, _ = image.shape

# Create an empty list to store frames
frames = []

while True:

    # Generate random coordinates for snowflakes
    snow_x = np.random.randint(0, cols, size=(100,))
    snow_y = np.random.randint(0, rows, size=(100,))

    # Overlay snowflakes on the image
    snow_image = image.copy()

    # Loop through the snowflakes and overlay them on the image
    for x, y in zip(snow_x, snow_y):
        snow_image[y, x] = [255, 255, 255]

    # Append the modified frame to the list
    frames.append(snow_image)

    # Display the modified image
    cv2.imshow("output", snow_image)

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

# Save the list of frames as a GIF
imageio.mimsave("snowfall.gif", frames)

# Close all windows
cv2.destroyAllWindows()
