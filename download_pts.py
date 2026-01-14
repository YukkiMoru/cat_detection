from ultralytics import YOLO

def main():
    print("モデルをダウンロードしています...")
    # モデルを読み込むだけでなければ自動的にダウンロードされます
    YOLO("yolov8n.pt")
    YOLO("yolov10n.pt")
    YOLO("yolo11n.pt")
    YOLO("yolo26n.pt")
    print("ダウンロード完了！")

if __name__ == "__main__":
    main()
