import cv2
import numpy as np

# Define the number of inner corners in each row and column
rows = 7
cols = 7

# Define the size of each square in the chessboard
square_size = 40

# Calculate the width and height of the chessboard
width = cols * square_size
height = rows * square_size

# Create an empty chessboard image
chessboard = np.zeros((height, width, 3), dtype=np.uint8)

# Fill the chessboard with alternating black and white squares
for i in range(0, height, square_size):
    for j in range(0, width, square_size):
        if (i // square_size + j // square_size) % 2 == 0:
                        chessboard[i:i+square_size, j:j+square_size] = (255, 255, 255)

# Display the chessboard
cv2.imwrite('Chessboard.jpg',chessboard)
cv2.waitKey()
