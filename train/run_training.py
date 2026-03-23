#!/usr/bin/env python
# uv run train/run_training.py
# yolo26n.pt & dataset_augment -(train)-> yolo26n.pt

from ultralytics import YOLO


def main():
    # 1. モデルの読み込み (yolo26n.pt)
    model = YOLO("yolo26n.pt")

    # 2. 学習の実行 (RTX 3070 をフル活用)
    model.train(
        data="train/data.yaml",  # データセットの定義
        epochs=100,  # 最大学習回数
        imgsz=640,  # 画像サイズ
        batch=16,  # VRAM 8GB向けのバッチサイズ（エラーが出たら8に下げる）
        device=0,  # RTX 3070 (0番GPU) を指定
        workers=4,  # CPUのデータ読み込みスレッド数
        project="train/runs",  # 結果の保存先フォルダ
        name="night_cat",  # 保存名
    )


if __name__ == "__main__":
    main()
