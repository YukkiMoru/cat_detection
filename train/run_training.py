#!/usr/bin/env python
from ultralytics import YOLO


def main():
    # 1. モデルの読み込み
    # ※事前学習済みの重み（yolo26n.pt等）をベースにします
    model = YOLO("yolo26n.pt")

    # 2. 学習の実行
    model.train(
        data="train/dataset_generated/data.yml",
        epochs=100,  # 最大学習回数
        imgsz=640,  # 画像サイズ
        # --- ハードウェア最適化 (Colab T4向け) ---
        batch=-1,  # AutoBatch: VRAMの限界を自動計算
        device=0,  # 0番GPU (T4) を指定
        workers=1,  # ColabのCPUコア数に合わせてボトルネックを防ぐ
        amp=True,  # 自動混合精度 (高速化・省メモリ)
        cache=True,  # RAMキャッシュ (※もしメモリ不足でColabが落ちる場合は False に変更してください)
        # --- 追加学習(ファインチューニング)のための最重要設定 ---
        lr0=0.01,  # 【必須】学習率を小さくし、賢いAIの既存の記憶が壊れるのを防ぐ
        freeze=0,  # 【必須】モデルの基礎部分（バックボーン）を凍結し、過学習を防止
        patience=20,  # 【必須】Valの精度が20エポック連続で上がらなければ、無駄な学習を早期終了
        # --- データ拡張の制御 (二重加工の防止) ---
        # 既に独自のスクリプトで背景を傾けたり合成したりしているため、
        # YOLO側での過剰な加工をオフにして、画像が原型を留めなくなるのを防ぎます。
        mosaic=0.0,  # 画像を4分割して繋ぎ合わせるモザイク処理をオフ
        degrees=0.0,  # 回転処理をオフ（Python側ですでに実装済みのため）
        # ---  保存設定 ---
        project="train/runs",  # 結果の保存先フォルダ
        name="night_cat",  # 今回の学習結果の名前
        exist_ok=True,  # 複数回実行時に上書き・継続
    )


if __name__ == "__main__":
    main()
