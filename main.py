import cv2
import time
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
cap = cv2.VideoCapture(0)
start_time = None
last_seen = 0
notified = False

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = model(frame, classes=[15], verbose=False)
    
    if len(results[0].boxes) > 0:
        last_seen = time.time()
        if start_time is None:
            start_time = time.time()
            notified = False
        elif time.time() - start_time >= 10 and not notified:
            print("検出")
            notified = True
    elif start_time is not None and time.time() - last_seen > 1.0:
        start_time = None
        notified = False

    cv2.imshow("YOLO11", results[0].plot())
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()