import os
import sys
import time

import cv2


def main():
    # 保存先ディレクトリの作成
    save_dir = os.path.join("dataset", "images")
    os.makedirs(save_dir, exist_ok=True)

    # カメラの設定
    CAMERA_ID = 0
    if sys.platform == "win32":
        backend = cv2.CAP_DSHOW
    elif sys.platform.startswith("linux"):
        backend = cv2.CAP_V4L2
    else:
        backend = cv2.CAP_ANY

    cap = cv2.VideoCapture(CAMERA_ID, backend)
    if not cap.isOpened():
        print("カメラが開けません")
        return

    # 解像度設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("====================================")
    print("データセット収集ツール起動")
    print(f"保存先: {save_dir}")
    print("[Space]キー / [Enter]キー: 写真を撮影")
    print("[Q]キー / [Esc]キー: 終了")
    print("====================================")

    count = 0
    # 既存のファイルから開始番号を決定
    existing_files = [
        f for f in os.listdir(save_dir) if f.startswith("img_") and f.endswith(".jpg")
    ]
    if existing_files:
        indices = [int(f.split("_")[1].split(".")[0]) for f in existing_files]
        count = max(indices) + 1

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # 画面表示
            cv2.imshow("Dataset Collector", frame)

            # キー入力待ち
            key = cv2.waitKey(1) & 0xFF

            # スペースキーまたはエンターキーで撮影
            if key == ord(" ") or key == 13:
                filename = os.path.join(save_dir, f"img_{count:04d}.jpg")
                cv2.imwrite(filename, frame)
                print(f"[{count:04d}] 保存しました: {filename}")

                count += 1

            # QキーまたはEscキーで終了
            elif key == ord("q") or key == 27:
                print("撮影を終了します。")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
