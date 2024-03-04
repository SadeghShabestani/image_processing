import cv2

capture = cv2.VideoCapture(0)

# Load the stickers
face_sticker = cv2.imread("face_imoje.png")  
glasses_sticker = cv2.imread("glasses.png")
smile_sticker = cv2.imread("smile.png")

# Load the Haar cascades
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_detector = cv2.CascadeClassifier("haarcascade_eye.xml")
smile_detector = cv2.CascadeClassifier("haarcascade_smile.xml")

sticker_mode = 0  # Initial mode: No sticker

while True:
    _, frame = capture.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if sticker_mode == 1:
        faces = face_detector.detectMultiScale(frame_gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_sticker = cv2.resize(face_sticker, (w, h))
            frame[y: y + h, x:x + w] = face_sticker

    if sticker_mode == 2:
        eyes = eye_detector.detectMultiScale(frame_gray, 1.3, 5)
        if len(eyes) >= 2:
            # Calculate bounding box for both eyes
            x_min = min(eyes[:,0])
            x_max = max(eyes[:,0] + eyes[:,2])
            y_min = min(eyes[:,1])
            y_max = max(eyes[:,1] + eyes[:,3])
            w = x_max - x_min
            h = y_max - y_min
            # Resize and apply glasses sticker
            resized_glasses_sticker = cv2.resize(glasses_sticker, (w, h))
            frame[y_min: y_min + h, x_min:x_min + w] = resized_glasses_sticker

        smile = smile_detector.detectMultiScale(frame, 1.7, 22)
        for (x, y, w, h) in smile:
            smile_sticker = cv2.resize(smile_sticker, (w, h))
            frame[y: y + h, x:x + w] = smile_sticker

    cv2.imshow("result", frame)

    key = cv2.waitKey(25) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('1'):
        sticker_mode = 1 
    elif key == ord('2'):
        sticker_mode = 2