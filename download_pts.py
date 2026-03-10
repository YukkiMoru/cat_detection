from ultralytics import YOLO


def main():
    print("モデルをダウンロードしています...")
    # モデルを読み込むだけでなければ自動的にダウンロードされます
    # YOLO("yolov8n.pt")
    # YOLO("yolov10n.pt")
    # YOLO("yolo11n.pt")
    # YOLO("yolo26n.pt")

    model = YOLO("yolo26n.pt")
    # Raspberry Pi用に推論サイズを320x320に固定してエクスポートします
    model.export(format="onnx", imgsz=320)
    print("ダウンロード完了！")


if __name__ == "__main__":
    main()
