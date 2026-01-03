import cv2, time, requests, threading, yaml
from datetime import datetime, timedelta, timezone
from ultralytics import YOLO

try:
    with open("config.yml", "r", encoding="utf-8") as f: CFG = yaml.safe_load(f)
except: CFG = {}

def get(path, default):
    val = CFG
    for key in path.split('.'):
        val = val.get(key) if isinstance(val, dict) else None
    return val if val is not None else default

WEBHOOK_FILE = get('discord.webhook_url_file', ".secrets/DWU")
try:
    with open(WEBHOOK_FILE, "r") as f: WEBHOOK_URL = f.read().strip()
except: WEBHOOK_URL = ""

HEADLESS = get('app.headless', False)
FRAME_SIZE = (get('camera.width', 320), get('camera.height', 240))
FPS = get('camera.fps', 2)
THRESH_DUR = get('detection.duration_threshold', 3.0)
THRESH_RESET = get('detection.reset_threshold', 1.0)
CONF = get('detection.confidence', 0.4)
CLASS_ID = get('detection.class_id', 15)

def notify(msg):
    if WEBHOOK_URL.startswith("http"):
        try: requests.post(WEBHOOK_URL, json={"content": msg}, timeout=10)
        except: pass

def main():
    model = YOLO('yolo11n.pt')
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
    cap.set(cv2.CAP_PROP_FPS, FPS)

    start_time, last_seen, notified = None, 0, False
    print("Monitoring...")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            results = model(frame, classes=[CLASS_ID], conf=CONF, verbose=False)
            now = time.time()
            
            if len(results[0].boxes):
                last_seen = now
                if start_time is None: start_time = now
                
                if not notified and (now - start_time >= THRESH_DUR):
                    print("Detected!")
                    ts = datetime.now(timezone(timedelta(hours=9), 'JST')).strftime("%Y/%m/%d %H:%M:%S")
                    threading.Thread(target=notify, args=(f"@everyone [{ts}] 猫を検出しました！🐈",), daemon=True).start()
                    notified = True
            elif start_time and (now - last_seen > THRESH_RESET):
                start_time, notified = None, False

            if not HEADLESS:
                cv2.imshow("YOLO11", results[0].plot())
                if cv2.waitKey(1) & 0xFF == ord("q"): break
    finally:
        cap.release()
        if not HEADLESS: cv2.destroyAllWindows()

if __name__ == "__main__": main()