import cv2


def list_available_cameras(max_check=10):
    """
    利用可能なカメラのインデックスとバックエンドを確認してリストアップする
    """
    print(f"Checking first {max_check} indexes for cameras...")
    available_cameras = []

    for index in range(max_check):
        # カメラを開く試行
        cap = cv2.VideoCapture(index)

        if cap.isOpened():
            # 実際にフレームが読めるか確認
            ret, _ = cap.read()
            if ret:
                # 解像度の取得
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                print(f"[OK] Camera Index {index}: Resolution={w}x{h}, FPS={fps}")
                available_cameras.append(index)
            else:
                print(f"[Warning] Camera Index {index}: Opened but failed to read frame.")

            cap.release()
        else:
            # 開けなかった場合は何もしない（通常はここを通る）
            pass

    print("-" * 30)
    if available_cameras:
        print(f"利用可能なカメラID: {available_cameras}")
    else:
        print("利用可能なカメラが見つかりませんでした。")
        print("接続を確認するか、libcameraコマンドを試してください。")

if __name__ == "__main__":
    list_available_cameras()
