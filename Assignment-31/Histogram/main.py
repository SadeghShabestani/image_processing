import cv2
import matplotlib.pyplot as plt


def histogram(image):
    # Initialize a list of zeros with a length of 256 (for each pixel value from 0 to 255)
    hist = [0] * 256
    # Iterate through each pixel in the image
    for row in image:
        for pixel in row:
            # Increment the histogram value for the pixel intensity
            hist[pixel] += 1
    return hist


# Read the image in grayscale
image = cv2.imread(
    "image_processing/Assignment-31/Histogram/input/medium.webp", cv2.IMREAD_GRAYSCALE
)

# Get the histogram as a list
histogram_instance = histogram(image=image)

# Plot the histogram using plot.plot()
plt.figure(figsize=(10, 4))
plt.plot(histogram_instance, color="blue")
plt.title("Histogram using plot.plot()")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.savefig("image_processing/Assignment-31/Histogram/output/histogram_plot.png")
plt.show()

# Plot the histogram using plt.hist()
plt.figure(figsize=(10, 4))
plt.hist(range(256), bins=256, weights=histogram_instance, color="green")
plt.title("Histogram using plt.hist()")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.savefig("image_processing/Assignment-31/Histogram/output/histogram_hist.png")
plt.show()

# Plot the histogram using plt.bar()
plt.figure(figsize=(10, 4))
plt.bar(range(256), histogram_instance, color="red")
plt.title("Histogram using plt.bar()")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.savefig("image_processing/Assignment-31/Histogram/output/histogram_bar.png")
plt.show()
