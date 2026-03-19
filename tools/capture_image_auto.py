#!/usr/bin/env python
# uv run tools/standalone_collector.py

import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

# ==========================================
# ⚙️ 設定 (ここだけ環境に合わせて変更してください)
# ==========================================
MODEL_PATH = "yolo26n.pt"  # 収集用に使う軽いモデル (自動でダウンロードされます)
CAMERA_ID = 0  # カメラの番号
SAVE_INTERVAL = 1.0  # 何秒ごとに保存するか
DETECT_CLASS_ID = 15  # COCOデータセットの「猫」のIDは15
SAVE_CLASS_ID = 0  # 新しく作る専用モデルのIDは「0」にする
CONFIDENCE = 0.1  # 検出の閾値（0.4くらいが誤検知少なめ）

DATASET_DIR = Path("dataset_captured/1/train")
IMG_DIR = DATASET_DIR / "images"
LBL_DIR = DATASET_DIR / "labels"
# ==========================================


def main():
    # フォルダの作成
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    LBL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"🤖 モデル '{MODEL_PATH}' を準備しています...")
    # inference.pyを使わず、UltralyticsのYOLOを直接呼び出す（超シンプル）
    model = YOLO(MODEL_PATH)

    print("📷 カメラを起動しています...")
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_ANY)

    # 高画質(1080p)設定
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("❌ カメラが開けません")
        return

    # 画面サイズを調整して表示
    cv2.namedWindow("Standalone Collector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Standalone Collector", 960, 540)

    last_save_time = 0
    print(f"✅ 監視スタート！ {SAVE_INTERVAL}秒ごとに自動保存します。(Qキーで終了)")

    try:
        while cap.isOpened():
            # カメラの遅延（ラグ）を防ぐための読み捨て
            for _ in range(2):
                cap.grab()
            ret, frame = cap.read()
            if not ret:
                continue

            # 推論の実行（対象を猫=15 に絞る）
            results = model(
                frame, classes=[DETECT_CLASS_ID], conf=CONFIDENCE, verbose=False
            )
            now = time.time()

            # 1つでも検出されていればTrue
            detected = len(results[0].boxes) > 0

            # 猫がいて、かつ設定した秒数が経過していたら保存
            if detected and (now - last_save_time > SAVE_INTERVAL):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                img_path = IMG_DIR / f"cat_{ts}.jpg"
                txt_path = LBL_DIR / f"cat_{ts}.txt"

                # 1. 画像の保存 (前処理なしの生画像を保存)
                cv2.imwrite(str(img_path), frame)

                # 2. ラベルの保存
                with open(txt_path, "w") as f:
                    for box in results[0].boxes:
                        # YOLO形式 (center_x center_y width height)
                        xywh = box.xywhn[0].tolist()
                        # 新しいデータセット用に ID=0 として記録
                        f.write(
                            f"{SAVE_CLASS_ID} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}\n"
                        )

                print(f"📸 保存しました: {img_path.name}")
                last_save_time = now

            # 画面表示 (推論結果の枠を描画)
            display_frame = results[0].plot() if detected else frame

            # 👇ここを追加！：GUI表示用だけ画像を半分のサイズ(960x540)に縮小する
            gui_frame = cv2.resize(display_frame, (960, 540))
            cv2.imshow("Standalone Collector", gui_frame)

            # Qキーで終了
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("🛑 終了しました")


if __name__ == "__main__":
    main()
