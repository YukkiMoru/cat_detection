#!/usr/bin/env python
# uv run train/run_training.py
# train/yolo26n.pt & dataset_augment -(train)-> train/runs/night_cat/weights/best.pt

from ultralytics import YOLO


def main():
    model = YOLO("train/models_origin/yolo26n.pt")

    model.train(
        # ここを絶対パスに変更！
        # data="/content/drive/MyDrive/ColabWorks/cat_detection/train/data.yaml",
        data="train/data.yaml",
        epochs=30,
        imgsz=640,
        batch=-1,
        device=0,
        workers=12,
        # project="/content/drive/MyDrive/ColabWorks/cat_detection/train/runs",
        project="train/runs",
        name="night_cat",
    )


if __name__ == "__main__":
    main()
