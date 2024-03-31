import cv2
import numpy as np

# Importing face detection and alignment models
from TFLiteFaceDetector import UltraLightFaceDetecion
from TFLiteFaceAlignment import CoordinateAlignmentModel

# Initializing face detection and alignment models
fd = UltraLightFaceDetecion("Assignment-30/rotateـphoto/weights/RFB-320.tflite", conf_threshold=0.88)
fa = CoordinateAlignmentModel("Assignment-30/rotateـphoto/weights/coor_2d106.tflite")

# Function to recognize facial features and extract landmarks
def recognition_of_facial_features(image, index, apple):
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
        big_mask = cv2.resize(multi, (0, 0), fx=2, fy=2)

        # Replace empty areas in the mask with apple
        for row in range(2*h):
            for col in range(2*w):
                if all(big_mask[row][col] == [0, 0, 0]):  # Check if pixel is empty
                    big_mask[row][col] = apple[y - h // 2 + row, x - w // 2 + col]

        # Adjust size of regions for pasting back into apple        
        if h % 2 == 0 :
            h_high = (3 * h // 2) 
        
        else :
            h_high = (3 * h // 2) + 1

        if w % 2 == 0 :
            w_high = (3 * w // 2)

        else:
            w_high = (3 * w // 2) + 1
        
        # Paste the modified mask back into the apple
        apple [y - h // 2 : y + h_high , x - w // 2 : x + w_high] = big_mask
    
    return apple

# Read the input images
image = cv2.imread("Assignment-30/snapchat_filter/input/face.png")
image = cv2.resize(image, (600, 600))
apple = cv2.imread("Assignment-30/snapchat_filter/input/apple.png")
apple = cv2.resize(apple, (image.shape[0], image.shape[1]))

# Indices for facial features
lips_landmarks = [52, 55, 56, 53, 59, 58, 61, 68, 67, 70, 67, 71, 63, 64]
eye_right = [39, 42, 40, 41, 35, 36, 33, 37]
eye_left = [89, 90, 87, 91, 93, 96, 94, 95]

# Recognizing facial features and extracting landmarks
res = recognition_of_facial_features(image=image, index=lips_landmarks, apple=apple)
res = recognition_of_facial_features(image=image, index=eye_right, apple=apple)
res = recognition_of_facial_features(image=image, index=eye_left, apple=apple)

# Display and save the result
cv2.imshow("result", res)
cv2.waitKey()
cv2.imwrite("Assignment-30/snapchat_filter/output/result.png", res)
