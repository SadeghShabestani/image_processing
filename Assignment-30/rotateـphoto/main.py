import cv2
import numpy as np

# Importing face detection and alignment models
from TFLiteFaceDetector import UltraLightFaceDetecion
from TFLiteFaceAlignment import CoordinateAlignmentModel

# Initializing face detection and alignment models
fd = UltraLightFaceDetecion("Assignment-30/rotateـphoto/weights/RFB-320.tflite", conf_threshold=0.88)
fa = CoordinateAlignmentModel("Assignment-30/rotateـphoto/weights/coor_2d106.tflite")

# Function to recognize facial features and extract landmarks
def recognition_of_facial_features(image, index):
    boxes, ـ = fd.inference(image)
    for pred in fa.get_landmarks(image, boxes):
        landmarks = []
        for i in index:
            landmarks.append(pred[i])
        landmarks = np.array(landmarks, dtype=int)

        x, y, w, h = cv2.boundingRect(landmarks)
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [landmarks], -1, (255, 255, 255), -1)
        mask = mask // 255
        multi = cv2.multiply(image, mask)
        multi = multi[y:y+h, x:x+w]

        rotate_mask = cv2.flip(multi, 0)
        for row in range(h):
            for col in range(w):
                if all(rotate_mask[row][col] == [0, 0, 0]):
                    rotate_mask[row][col] = image[y + row , x + col]

        image[y:y+h, x:x+w] = rotate_mask
    return image

# Reading the input image
image = cv2.imread("Assignment-30/rotateـphoto/input/face.jpeg")

# Indices for facial features
lips_landmarks = [52, 55, 56, 53, 59, 58, 61, 68, 67, 70, 67, 71, 63, 64]
eye_right = [39, 42, 40, 41, 35, 36, 33, 37]
eye_left = [89, 90, 87, 91, 93, 96, 94, 95]

# Recognizing facial features and extracting landmarks
res = recognition_of_facial_features(image=image, index=lips_landmarks)
res = recognition_of_facial_features(image=image, index=eye_right)
res = recognition_of_facial_features(image=image, index=eye_left)

# Flipping the final result
result = cv2.flip (res , 0)

# Displaying and saving the result
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.imwrite("Assignment-30/rotateـphoto/output/result.jpeg", result)