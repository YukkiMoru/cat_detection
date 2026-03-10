from ultralytics import YOLO


def main():
    print("モデルをダウンロードしています...")
    # モデルを読み込むだけでなければ自動的にダウンロードされます
    # YOLO("models/yolov8n.pt")
    # YOLO("models/yolov10n.pt")
    # YOLO("models/yolo11n.pt")
    # YOLO("models/yolo26n.pt")

    model = YOLO("models/yolo26n.pt")
    # Raspberry Pi用に推論サイズを320x320に固定してエクスポートします
    model.export(format="onnx", imgsz=640)
    print("ダウンロード完了！")


if __name__ == "__main__":
    main()
