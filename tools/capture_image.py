#!/usr/bin/env python
# uv run tools/capture_dataset.py

import os
import sys

import cv2


def main():
    # ベースディレクトリの設定
    base_dir = "dataset"
    # サブディレクトリ（ラベル名）の定義
    classes = {"cat": "猫あり", "no_cat": "猫なし"}

    for label in classes.keys():
        os.makedirs(os.path.join(base_dir, label), exist_ok=True)

    # カメラの設定
    CAMERA_ID = 0
    backend = cv2.CAP_ANY
    if sys.platform == "win32":
        backend = cv2.CAP_DSHOW
    elif sys.platform.startswith("linux"):
        backend = cv2.CAP_V4L2

    cap = cv2.VideoCapture(CAMERA_ID, backend)
    if not cap.isOpened():
        print("カメラが開けません")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print("====================================")
    print("🐾 猫データセット収集ツール 🐾")
    print(f"保存先: {base_dir}/")
    print("[C]キー: 「猫あり(cat)」として保存")
    print("[N]キー: 「猫なし(no_cat)」として保存")
    print("[Q]キー: 終了")
    print("====================================")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            # プレビュー表示
            cv2.imshow("Dataset Collector", frame)
            key = cv2.waitKey(1) & 0xFF

            # 保存処理の関数化（カウントを自動計算して保存）
            def save_image(label_name):
                target_dir = os.path.join(base_dir, label_name)
                # 既存ファイル数から番号を決定
                existing_count = len(
                    [f for f in os.listdir(target_dir) if f.endswith(".jpg")]
                )
                filename = os.path.join(
                    target_dir, f"{label_name}_{existing_count:04d}.jpg"
                )
                cv2.imwrite(filename, frame)
                print(f"✅ 【{classes[label_name]}】保存完了: {filename}")

            if key == ord("c"):  # Cat
                save_image("cat")
            elif key == ord("n"):  # No Cat
                save_image("no_cat")
            elif key == ord("q") or key == 27:
                print("撮影を終了します。")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
