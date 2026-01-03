import cv2, time, requests, threading, yaml
from datetime import datetime, timedelta, timezone
from ultralytics import YOLO

try: C = yaml.safe_load(open("config.yml", "r", encoding="utf-8"))
except: C = {}
def g(k, d):
    v = C
    for x in k.split('.'): v = v.get(x) if isinstance(v, dict) else None
    return v if v is not None else d

WH = g('discord.webhook_url_file', ".secrets/DWU")
try: WH_URL = open(WH).read().strip()
except: WH_URL = ""

HL = g('app.headless', False)
SZ, FPS = (g('camera.width', 320), g('camera.height', 240)), g('camera.fps', 2)
DUR, RST, CONF, CID = g('detection.duration_threshold', 3.0), g('detection.reset_threshold', 1.0), g('detection.confidence', 0.4), g('detection.class_id', 15)
USE_M, M_TH, DBG_M = g('detection.use_motion_filter', True), g('detection.motion_threshold', 500), g('detection.debug_motion', False)

def notify():
    if WH_URL.startswith("http"):
        ts = datetime.now(timezone(timedelta(hours=9))).strftime('%Y/%m/%d %H:%M:%S')
        try: requests.post(WH_URL, json={"content": f"@everyone [{ts}] 猫検出！🐈"}, timeout=10)
        except: pass

def main():
    model = YOLO('yolo11n.pt')
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, SZ[0]); cap.set(4, SZ[1]); cap.set(5, FPS)
    st, ls, nt, pg = None, 0, False, None
    print("Monitoring...")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: time.sleep(0.1); continue

            run = True
            if USE_M:
                gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
                if pg is None: pg = gray; continue
                score = cv2.countNonZero(cv2.threshold(cv2.absdiff(pg, gray), 25, 255, cv2.THRESH_BINARY)[1])
                pg = gray
                if score < M_TH and st is None:
                    run = False
                    if DBG_M: print(f"Skip: {score}")

            res = model(frame, classes=[CID], conf=CONF, verbose=False) if run else []
            now = time.time()
            
            if run and res[0].boxes:
                ls = now
                if st is None: st = now
                if not nt and (now - st >= DUR):
                    print("Detected!"); threading.Thread(target=notify, daemon=True).start(); nt = True
            elif st and (now - ls > RST): st, nt = None, False

            if not HL:
                cv2.imshow("YOLO", res[0].plot() if run and res else frame)
                if cv2.waitKey(1) & 0xFF == ord("q"): break
            else: time.sleep(0.1)
    finally: cap.release(); cv2.destroyAllWindows() if not HL else None

if __name__ == "__main__": main()