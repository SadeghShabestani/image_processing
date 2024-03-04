import cv2
import numpy as np

# Create a white image
character = np.ones((600, 600), np.uint8) * 255

# Draw the letter T
cv2.rectangle(character, (150, 100), (450, 150), 0, -1)
cv2.rectangle(character, (300, 100), (350, 450), 0, -1)



# Save and display the image
cv2.imwrite('character_T.jpg', character)
cv2.waitKey()