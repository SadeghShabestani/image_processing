import cv2


def foreground_focus(image):

    # Apply a threshold to create a mask of the foreground
    _, mask = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)

    # Invert the mask to get the background
    mask_inv = cv2.bitwise_not(mask)

    # Blur the entire image
    blurred = cv2.GaussianBlur(image, (15, 15), 0)

    # Use the mask to keep the original flower region, and blur the background
    flower = cv2.bitwise_and(image, image, mask=mask)
    background = cv2.bitwise_and(blurred, blurred, mask=mask_inv)

    # Combine the foreground (flower) with the blurred background
    result = cv2.add(flower, background)

    return result


# Read the image
image = cv2.imread(
    "image_processing/Assignment-31/foreground_focus/input/flower.png",
    cv2.IMREAD_GRAYSCALE,
)

# Process the image to focus on the foreground
foreground_focus_instance = foreground_focus(image=image)

# Display the result
cv2.imshow("result", foreground_focus_instance)
cv2.waitKey(0)
cv2.imwrite(
    "image_processing/Assignment-31/foreground_focus/output/flower_result.png",
    foreground_focus_instance,
)
