import cv2

# Read the images for background, foreground, and the mask
room_background = cv2.imread('virtual_decoration/input/room_background.jpg')
room_foreground = cv2.imread('virtual_decoration/input/room_foreground.jpg')
room_mask = cv2.imread('virtual_decoration/input/room_mask.jpg')

# Normalize the mask to be in the range of [0, 1]
normalized_mask = room_mask / 255.0

# Apply the mask to the foreground by multiplying it, this will "cut out" the portion of the foreground
result_room_foreground = room_foreground * normalized_mask

# Create an inverted mask (areas that were black become white and vice versa)
mask_inverted = 255 - room_mask

# Normalize the inverted mask
normalized_mask_inverted = mask_inverted / 255.0

# Apply the inverted mask to the background, this will retain the background where the foreground isn't applied
result_room_background = room_background * normalized_mask_inverted

# Combine the masked foreground and background to create the composite image
result = result_room_foreground + result_room_background

# Write the composite image to a file in the output directory
cv2.imwrite("virtual_decoration/output/result.jpg", result)