import cv2

# Load the image
image = cv2.imread("Kitten-Growth-Stages.jpg")

# Load the cat face cascade
cat_face_cascade = cv2.CascadeClassifier("haarcascade_frontalcatface_extended.xml")

# Detect cats in the image
cats = cat_face_cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=3)

# Initialize the count of detected cats
count = 0
for (x, y, w, h) in cats:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    count += 1

# Put the count of cats on the image
cv2.putText(image, f"Number of cats: {count}", (20, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)

# Save the modified image
cv2.imwrite("cat_count.jpg", image)

cv2.imshow('output', image)
cv2.waitKey()