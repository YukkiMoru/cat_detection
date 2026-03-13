#!/usr/bin/env python
# uv run tools/capture_dataset.py
# internet -> models/yolo26n.pt, yolo26s.pt -> models/yolo26n/*, yolo26s/*

import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    # --- 設定 ---
    models_dir = Path("models")

    # キャッシュ（退避用）フォルダの作成
    cache_dir = models_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # エクスポート対象のベースYOLOモデル
    base_models = [
        "yolo26n.pt",  # Nano
        "yolo26s.pt",  # Small
        # "yolo26m.pt",  # Medium
        # "yolo26l.pt",  # Large
        # "yolo26x.pt",  # Extra Large
    ]
    image_sizes = [640, 320]

    # INT8キャリブレーション用のデータセット
    # 指定しないとUltralyticsのデフォルト(coco8.yaml)が使われます
    dataset_yaml = "coco8.yaml"

    # エクスポート設定の全パターン
    export_configs = [
        {"format": "mnn", "suffix": "mnn_fp32"},
        {"format": "mnn", "suffix": "mnn_fp16", "half": True},
        {"format": "mnn", "suffix": "mnn_int8", "int8": True, "data": dataset_yaml},
    ]

    total_exports = len(base_models) * len(image_sizes) * len(export_configs)
    current_count = 0

    print(f"=== 全{total_exports}パターンのエクスポートを開始します ===")

    for base_model_name in base_models:
        original_pt_path = models_dir / base_model_name
        model_prefix = original_pt_path.stem  # 'yolo26n' や 'yolo26s'

        # 直接モデル名のフォルダを作成
        model_output_dir = models_dir / model_prefix
        model_output_dir.mkdir(parents=True, exist_ok=True)

        # ベースモデルが存在しない場合は自動ダウンロードさせるため、一度ロードしておく
        if not original_pt_path.exists():
            print(f"{base_model_name} をダウンロード/準備中...")
            YOLO(str(original_pt_path))

        for imgsz in image_sizes:
            for config in export_configs:
                current_count += 1

                target_name = f"{model_prefix}_size{imgsz}_{config['suffix']}"
                temp_pt_path = model_output_dir / f"{target_name}.pt"

                print(
                    f"\n[{current_count}/{total_exports}] 作成中: {model_prefix} -> {target_name} ..."
                )

                # 既にエクスポート済みの目的のファイルがあるか確認
                target_ext = f".{config['format']}"  # 例: .mnn
                existing_outputs = list(
                    model_output_dir.glob(f"{target_name}{target_ext}")
                )
                if existing_outputs:
                    print(
                        f"  [スキップ] 既に出力があります: {existing_outputs[0].name}"
                    )
                    continue

                try:
                    # 1. 元の.ptファイルを、目的の名前でコピーする
                    shutil.copy(original_pt_path, temp_pt_path)

                    # 2. コピーしたモデルをロード
                    model = YOLO(str(temp_pt_path))

                    # 3. 引数を組み立ててエクスポート
                    export_args = {
                        "format": config["format"],
                        "imgsz": imgsz,
                        "half": config.get("half", False),
                        "int8": config.get("int8", False),
                    }
                    if "data" in config:
                        export_args["data"] = config["data"]

                    # 実行！
                    model.export(**export_args)

                except Exception as e:
                    print(f"  [エラー] {target_name} のエクスポートに失敗しました: {e}")

                finally:
                    # 4. 中間ファイル(.ptや.onnx等)を .cache に移動
                    for generated_file in model_output_dir.glob(f"{target_name}.*"):
                        # 万が一、大元の base_model (例: models/yolo26n.pt) と一致した場合はスキップ
                        if generated_file.absolute() == original_pt_path.absolute():
                            continue

                        # 最終的に欲しい拡張子（例: .mnn）以外は .cache フォルダへ移動
                        if generated_file.suffix != target_ext:
                            try:
                                dest_path = cache_dir / generated_file.name
                                # 同名ファイルがキャッシュ内に存在する場合は上書きのため事前削除
                                if dest_path.exists():
                                    if dest_path.is_dir():
                                        shutil.rmtree(dest_path)
                                    else:
                                        dest_path.unlink()

                                # .cache へ移動
                                shutil.move(str(generated_file), str(dest_path))
                            except Exception as e:
                                print(
                                    f"  [警告] 中間ファイルのキャッシュ移動に失敗しました: {generated_file.name} ({e})"
                                )

    print("\n=== すべてのエクスポート処理が完了しました！ ===")
    print("目的のモデル: 各モデルのフォルダ内")
    print(f"中間ファイル等: {cache_dir.absolute()} に退避しました")


if __name__ == "__main__":
    main()
