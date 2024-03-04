import cv2
import numpy as np

# Open the video capture device (webcam)
capture = cv2.VideoCapture(0)

# Read a frame from the capture
_, frame = capture.read()

# Get the dimensions of the frame
rows = frame.shape[0]
cols = frame.shape[1]

# Create a VideoWriter object to save the output video
writer = cv2.VideoWriter(
    "color.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (cols, rows)
)

while True:
    # Read a frame from the capture
    _, frame = capture.read()

    # Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get the dimensions of the grayscale frame
    rows, cols = frame_gray.shape

    # Define the region of interest (ROI)
    roi = frame[rows // 2 - 100 : cols // 2 + 100, cols // 2 - 100 : cols // 2 + 100]

    # Calculate the average pixel value of the ROI
    average = np.average(roi)

    # Determine the color label based on the average pixel value
    if average >= 130:
        color_label = "White"
    elif 80 <= average <= 129:
        color_label = "Gray"
    else:
        color_label = "Black"

    # Add text label to the frame based on the color detected
    cv2.putText(
        frame,
        color_label,
        (rows // 2 + 5, cols // 2 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )

    # Apply blur to the frame
    frame_blur = cv2.blur(frame, (40, 40))

    # Replace the ROI with the blurred ROI
    frame_blur[rows // 2 - 100 : cols // 2 + 100, cols // 2 - 100 : cols // 2 + 100] = (
        roi
    )

    # Write the modified frame to the output video
    writer.write(frame_blur)

    # Display the modified frame
    cv2.imshow("output", frame_blur)

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

# Release the VideoWriter object
writer.release()

# Close all windows
cv2.destroyAllWindows()
