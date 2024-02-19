import cv2

width = 700
height = 500
capture = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier("hand.xml")

while True:
    ret, frame = capture.read()
    frame = cv2.resize(frame, (width, height))
    frame = cv2.flip(frame, 1)
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    results = detector.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=10)

    for (x, y, w, h) in results: 
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

    cv2.imshow("Capture Window", frame)
    # Check if user wants to exit.
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()








