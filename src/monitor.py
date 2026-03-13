import logging
import signal
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import requests

import config
from inference import CatDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
running = True


def get_webhook_url():
    """Webhook URLをファイルから読み込む"""
    try:
        return (
            Path(".secrets") / "DWU"
            and Path(".secrets/DWU").read_text(encoding="utf-8").strip()
        )
    except Exception as e:
        logging.debug(f"Webhook読み込みエラー: {e}")
        return ""


def send_notification(url, frame):
    """画像付きでDiscordに通知を送る"""
    if not url or not url.startswith("http"):
        return

    success, img = cv2.imencode(".jpg", frame)
    if not success:
        return

    files = {"file": ("cat.jpg", img.tobytes(), "image/jpeg")}
    ts = datetime.now(timezone(timedelta(hours=9))).strftime("%H:%M:%S")
    data = {"content": f"[{ts}] 猫検出！🐈"}

    def _post():
        try:
            requests.post(url, data=data, files=files, timeout=10)
        except Exception as e:
            logging.error(f"通知エラー: {e}")

    threading.Thread(target=_post, daemon=True).start()


def signal_handler():
    global running
    running = False


def main():
    global running
    signal.signal(signal.SIGINT, signal_handler)

    logging.info("起動中...")
    webhook_url = get_webhook_url()

    # モデルとカメラの準備
    try:
        detector = CatDetector()
    except Exception as e:
        logging.error(f"モデルが見つかりません: {e}")
        return

    if sys.platform == "win32":
        backend = cv2.CAP_DSHOW
    elif sys.platform.startswith("linux"):
        backend = cv2.CAP_V4L2
    else:
        backend = cv2.CAP_ANY

    cap = cv2.VideoCapture(config.CAMERA_ID, backend)
    if not cap.isOpened():
        logging.error("カメラが開けません")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 状態管理用変数
    start_time = None
    last_seen = 0
    notified = False
    best_frame = None
    max_conf = 0.0
    target_interval = 1.0 / config.FPS

    logging.info("監視開始 (Ctrl+Cで停止)")

    try:
        while running and cap.isOpened():
            loop_start = time.time()

            # カメラのラグを防ぐため古いフレームを読み捨てる
            for _ in range(5):
                cap.grab()
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            detected, current_conf, results = detector.detect_cat(frame)

            now = time.time()

            # 判定ロジック
            if detected:
                last_seen = now

                # ベストショット更新
                if current_conf > max_conf:
                    max_conf = current_conf
                    best_frame = frame.copy()

                if start_time is None:
                    start_time = now
                    logging.info(f"猫検出開始 (信頼度: {current_conf:.2f})")

                # 一定時間継続したら通知
                if not notified and (now - start_time >= config.DURATION_THRESH):
                    logging.info("通知送信！")
                    img_to_send = best_frame if best_frame is not None else frame
                    send_notification(webhook_url, img_to_send)
                    notified = True

            # 見失って一定時間経過でリセット
            elif start_time and (now - last_seen > config.RESET_THRESH):
                logging.info("リセット")
                start_time = None
                notified = False
                best_frame = None
                max_conf = 0.0

            # 表示と待機
            wait_time = target_interval - (time.time() - loop_start)

            if not config.HEADLESS:
                img = results[0].plot() if detected and results else frame
                cv2.imshow("Cat", img)
                if cv2.waitKey(int(max(1, wait_time * 1000))) & 0xFF == ord("q"):
                    break
            else:
                if wait_time > 0:
                    time.sleep(wait_time)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
