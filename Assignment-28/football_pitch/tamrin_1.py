import cv2
import numpy as np

def create_soccer_field_pattern(height, width):
    """
    Creates an image simulating a soccer field with alternating green stripes.

    Parameters:
    - height: The height of the image in pixels.
    - width: The width of the image in pixels.

    Returns:
    - A numpy array representing the soccer field image.
    """
    field_pattern = np.zeros((height, width, 3), np.uint8)
    stripe_count = 8  # Define the total number of stripes
    for stripe_index in range(stripe_count):
        # Calculate stripe positions and width
        stripe_start = (width // stripe_count) * stripe_index
        stripe_end = stripe_start + (width // stripe_count)
        # Choose color based on the stripe's index
        stripe_color = [65, 130, 20] if stripe_index % 2 == 0 else [60, 120, 30]
        # Apply color to the stripe
        field_pattern[:, stripe_start:stripe_end] = stripe_color

    # Draw a white rectangle around the field for the boundary
    cv2.rectangle(field_pattern, (20, 20), (width - 20, height - 20), (255, 255, 255), 2)

    # Draw a white line for the center line
    cv2.line(field_pattern, (width // 2, 20), (width // 2, height - 20), (255, 255, 255), 2)

    # Draw a white circle for the center circle
    cv2.circle(field_pattern, (width // 2, height // 2), 100, (255, 255, 255), 2) 

    # Goal dimensions and positions
    goal_width = width // 14
    goal_height = height // 6
    # Left goal
    cv2.rectangle(field_pattern, (20, height // 2 - goal_height // 2), (20 + goal_width, height // 2 + goal_height // 2), (255, 255, 255), 2)
    # Right goal
    cv2.rectangle(field_pattern, (width - 20 - goal_width, height // 2 - goal_height // 2), (width - 20, height // 2 + goal_height // 2), (255, 255, 255), 2)

    return field_pattern

# Set the dimensions for the soccer field image
img_height = 600
img_width = 800

# Generate the soccer field pattern
soccer_field_img = create_soccer_field_pattern(img_height, img_width)

# Display the generated soccer field pattern
cv2.imshow("Custom Soccer Field", soccer_field_img)
cv2.imwrite("custom_soccer_field.jpg", soccer_field_img)
cv2.waitKey(0)
cv2.destroyAllWindows()