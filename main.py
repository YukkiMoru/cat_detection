import cv2
import time
from ultralytics import YOLO

HEADLESS = False

model = YOLO('yolo11n.pt')
cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)

start_time = None
last_seen = 0
notified = False

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = model(frame, classes=[15], verbose=False)
    is_cat = len(results[0].boxes) > 0
    now = time.time()
    
    if is_cat:
        last_seen = now
        if start_time is None:
            start_time = now
            notified = False
        
        if not notified and (now - start_time >= 10):
            print("検出")
            notified = True
            
    elif start_time and (now - last_seen > 2.0):
        start_time = None

    if not HEADLESS:
        cv2.imshow("YOLO11", results[0].plot())
        if cv2.waitKey(1) & 0xFF == ord("q"): break
    
    time.sleep(0.2) # 負荷軽減(約5fps)

cap.release()
if not HEADLESS:
    cv2.destroyAllWindows()