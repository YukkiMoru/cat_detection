#!/usr/bin/env python
import datetime
import logging
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import requests

import config
from inference import CatDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
running = True

# --- 保存先ディレクトリの設定 ---
DATASET_ROOT = Path("dataset_yolo")
IMG_DIR = DATASET_ROOT / "train" / "images"
LBL_DIR = DATASET_ROOT / "train" / "labels"

for d in [IMG_DIR, LBL_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def get_webhook_url():
    """Webhook URLを読み込む"""
    try:
        p = Path(".secrets/DWU")
        return p.read_text(encoding="utf-8").strip() if p.exists() else ""
    except Exception as e:
        logging.debug(f"Webhook読み込みエラー: {e}")
        return ""


def save_for_yolo(frame, results):
    """検出時の画像とYOLO形式のラベルを保存"""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    img_path = IMG_DIR / f"auto_cat_{ts}.jpg"
    txt_path = LBL_DIR / f"auto_cat_{ts}.txt"

    # 画像保存（高画質1080pのまま保存）
    cv2.imwrite(str(img_path), frame)

    # YOLOラベル保存
    if results and len(results) > 0:
        with open(txt_path, "w") as f:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                # 正規化座標 (0.0 - 1.0) なので解像度によらず正確
                xywh = box.xywhn[0].tolist()
                f.write(
                    f"{cls} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}\n"
                )

    logging.info(f"💾 高画質データを保存しました: {img_path.name}")


def send_notification(url, frame):
    """Discord通知（画像は軽量化して送信）"""
    if not url or not url.startswith("http"):
        return

    # 通知用には少し縮小して送る（送信速度と容量のため）
    small_frame = cv2.resize(frame, (960, 540))
    success, img = cv2.imencode(".jpg", small_frame)
    if not success:
        return

    files = {"file": ("cat.jpg", img.tobytes(), "image/jpeg")}
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    data = {"content": f"[{ts}] 🐈 猫を検出！高画質画像を保存しました。"}

    def _post():
        try:
            requests.post(url, data=data, files=files, timeout=10)
        except Exception as e:
            logging.error(f"通知エラー: {e}")

    threading.Thread(target=_post, daemon=True).start()


def signal_handler(sig, frame):
    global running
    running = False


def main():
    global running
    signal.signal(signal.SIGINT, signal_handler)

    logging.info("起動中...")
    webhook_url = get_webhook_url()

    try:
        detector = CatDetector()
    except Exception as e:
        logging.error(f"モデルが見つかりません: {e}")
        return

    # カメラの初期化
    if sys.platform == "win32":
        backend = cv2.CAP_DSHOW
    elif sys.platform.startswith("linux"):
        backend = cv2.CAP_V4L2
    else:
        backend = cv2.CAP_ANY

    cap = cv2.VideoCapture(config.CAMERA_ID, backend)

    # --- 重要: 1920x1080 設定の黄金順序 ---
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        logging.error("カメラが開けません。解像度が未対応か、別のアプリが使用中です。")
        return

    # 設定された実際のサイズを確認
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    logging.info(f"カメラ解像度: {actual_w}x{actual_h} (MJPG)")

    # 表示ウィンドウを可変サイズにする（1080pだと画面からはみ出るため）
    if not config.HEADLESS:
        cv2.namedWindow("Cat Monitor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Cat Monitor", 960, 540)  # 表示だけ半分に縮小

    # 状態管理
    start_time = None
    last_seen = 0
    notified = False
    best_frame = None
    best_results = None
    max_conf = 0.0
    target_interval = 1.0 / config.FPS

    logging.info("監視開始 (QキーまたはCtrl+Cで停止)")

    try:
        while running and cap.isOpened():
            loop_start = time.time()

            # ラグ防止の読み捨て
            for _ in range(2):
                cap.grab()
            ret, frame = cap.read()
            if not ret:
                continue

            # 推論
            detected, current_conf, results = detector.detect_cat(frame)
            now = time.time()

            if detected:
                last_seen = now
                # ベストショットを更新
                if current_conf > max_conf:
                    max_conf = current_conf
                    best_frame = frame.copy()
                    best_results = results

                if start_time is None:
                    start_time = now
                    logging.info(f"猫を捕捉 (信頼度: {current_conf:.2f})")

                # 一定時間（config.DURATION_THRESH）映り続けたら保存＆通知
                if not notified and (now - start_time >= config.DURATION_THRESH):
                    save_frame = best_frame if best_frame is not None else frame
                    save_res = best_results if best_results is not None else results

                    save_for_yolo(save_frame, save_res)
                    # send_notification(webhook_url, save_frame)
                    notified = True

            elif start_time and (now - last_seen > config.RESET_THRESH):
                logging.info("リセット（猫がいなくなりました）")
                start_time = None
                notified = False
                best_frame = None
                best_results = None
                max_conf = 0.0

            # 描画処理
            if not config.HEADLESS:
                display_img = results[0].plot() if detected and results else frame
                cv2.imshow("Cat Monitor", display_img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # FPS調整
            wait_time = target_interval - (time.time() - loop_start)
            if wait_time > 0:
                time.sleep(wait_time)

    finally:
        logging.info("クリーンアップ中...")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
