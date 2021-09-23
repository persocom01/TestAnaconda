import cv2

# Try either VideoCapture(0) or (1) based on camera availability.
webcam = cv2.VideoCapture(0)

check, frame = webcam.read()
cv2.imwrite(filename=r'./images/webcam.jpg', img=frame)
webcam.release()
