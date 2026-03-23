#!/usr/bin/env python
# uv run tools/capture_dataset.py

import os
import sys
import time

import cv2


def main():
    # --- 設定 ---
    base_dir = "dataset_captured"
    classes = {"cat": "猫あり", "no_cat": "猫なし"}
    width, height = 1920, 1080  # 希望解像度

    # ディレクトリ作成
    for label in classes.keys():
        os.makedirs(os.path.join(base_dir, label), exist_ok=True)

    # --- カメラ初期化 ---
    CAMERA_ID = 0
    backend = cv2.CAP_ANY
    if sys.platform == "win32":
        backend = cv2.CAP_DSHOW
    elif sys.platform.startswith("linux"):
        backend = cv2.CAP_V4L2

    cap = cv2.VideoCapture(CAMERA_ID, backend)

    if not cap.isOpened():
        print("❌ カメラが開けません")
        return

    # --- 画質向上のための詳細設定 ---
    # 1. 解像度の指定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # 2. コーデックをMJPGに設定（USB帯域不足による解像度低下を防ぐ）
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    # 3. バッファサイズを1に設定（常に最新のフレームを取得し、タイムラグを減らす）
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 実際の解像度を確認
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("====================================")
    print("🐾 猫データセット収集ツール (High-Quality Mode) 🐾")
    print(f"設定解像度: {width}x{height}")
    print(f"実際の解像度: {actual_w}x{actual_h}")
    print(f"保存先: {base_dir}/")
    print("------------------------------------")
    print("💡 起動中... カメラの露出が安定するまで数秒お待ちください。")
    print("[C]キー: 「猫あり(cat)」として保存")
    print("[N]キー: 「猫なし(no_cat)」として保存")
    print("[Q]キー: 終了")
    print("====================================")

    # カメラのウォームアップ（露出調整待ち）
    time.sleep(2)

    try:
        while cap.isOpened():
            # バッファをクリアするために数回read()を回す手法もありますが、
            # 基本は直近の1フレームを取得します
            ret, frame = cap.read()
            if not ret:
                continue

            # 画面表示用のコピー（UIテキストを入れる場合はこれに描画）
            display_frame = frame.copy()
            cv2.putText(
                display_frame,
                f"{int(actual_w)}x{int(actual_h)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Dataset Collector (High-Res)", display_frame)
            key = cv2.waitKey(1) & 0xFF

            # --- 保存処理 ---
            def save_image(label_name):
                target_dir = os.path.join(base_dir, label_name)
                # 既存ファイル数から番号を決定 (.png)
                existing_count = len(
                    [f for f in os.listdir(target_dir) if f.lower().endswith(".png")]
                )
                filename = os.path.join(
                    target_dir, f"{label_name}_{existing_count:04d}.png"
                )

                # PNG形式で無圧縮（最高画質）保存
                # cv2.IMWRITE_PNG_COMPRESSION の範囲は 0-9 (0が低圧縮＝高品質・高速)
                success = cv2.imwrite(filename, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])

                if success:
                    print(f"✅ 【{classes[label_name]}】保存完了: {filename}")
                else:
                    print(f"❌ 保存に失敗しました: {filename}")

            if key == ord("c"):
                save_image("cat")
            elif key == ord("n"):
                save_image("no_cat")
            elif key == ord("q") or key == 27:
                print("撮影を終了します。")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
