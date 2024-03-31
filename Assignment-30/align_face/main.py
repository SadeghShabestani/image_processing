import cv2
import numpy as np

# Importing face detection and alignment models
from TFLiteFaceDetector import UltraLightFaceDetecion
from TFLiteFaceAlignment import CoordinateAlignmentModel

# Initializing face detection and alignment models
fd = UltraLightFaceDetecion("Assignment-30/align_face/weights/RFB-320.tflite", conf_threshold=0.88)
fa = CoordinateAlignmentModel("Assignment-30/align_face/weights/coor_2d106.tflite")

# Function to recognize facial features and extract landmarks
def recognition_of_facial_features(image, index):
    # Detecting faces in the image
    boxes, ـ = fd.inference(image)
    for pred in fa.get_landmarks(image, boxes):
        landmarks = []
        # Extracting specified landmarks
        for i in index:
            landmarks.append(pred[i])
        landmarks = np.array(landmarks, dtype=int)

    return landmarks

# Function to rotate and align the face based on eye positions
def rotate_align_face(image, lip, eye_r, eye_l):
    # Calculating the center of the left eye
    left_eye_center = np.mean(eye_l, axis=0).astype(int)
    # Calculating the center of the right eye
    right_eye_center = np.mean(eye_r, axis=0).astype(int)

    # Calculating the difference in y and x coordinates between the eyes
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]

    # Calculating the rotation angle based on the difference in coordinates
    angle = np.degrees(np.arctan2(dy, dx)) - 180

    # Defining the center of rotation as the center of the right eye
    eyes_center = (int(right_eye_center[0]), int(right_eye_center[1]))

    # Creating a rotation matrix to rotate around the eyes' center by the calculated angle
    rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale=1)

    # Applying the rotation to the original image
    aligned_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

    return aligned_image

# Reading the input image
image = cv2.imread("Assignment-30/align_face/input/face.jpeg")

# Indices for facial features
lips_landmarks = [52, 55, 56, 53, 59, 58, 61, 68, 67, 70, 67, 71, 63, 64]
eye_right = [39, 42, 40, 41, 35, 36, 33, 37]
eye_left = [89, 90, 87, 91, 93, 96, 94, 95]

# Recognizing facial features and extracting landmarks
lip = recognition_of_facial_features(image=image, index=lips_landmarks)
eye_r = recognition_of_facial_features(image=image, index=eye_right)
eye_l = recognition_of_facial_features(image=image, index=eye_left)

# Rotating and aligning the face based on eye positions
result_image = rotate_align_face(image, lip, eye_r, eye_l)

# Displaying the aligned image
cv2.imshow("result", result_image)
cv2.waitKey(0)
cv2.imwrite("Assignment-30/align_face/output/result.jpeg", result_image)