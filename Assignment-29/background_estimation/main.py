import cv2
import numpy as np

# Open Video
cap = cv2.VideoCapture('background_estimation/input/cars.mp4')

# Randomly select 25 frames
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)

# Store selected frames in an array
frames = []
for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    _, frame = cap.read()
    frames.append(frame)

# Calculate the median along the time axis
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)    

# Write the result to a file in the output directory.
cv2.imwrite("background_estimation/output/result.png", medianFrame)