import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    # --- 設定 ---
    models_dir = Path("models")
    output_dir = models_dir / "yolo26"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_models = ["yolo26n.pt", "yolo26s.pt"]
    image_sizes = [640, 416, 320]

    # INT8キャリブレーション用のデータセット（ご自身のデータセットyamlがあれば書き換えてください）
    # 指定しないとUltralyticsのデフォルト(coco8.yaml)が使われます
    dataset_yaml = "coco8.yaml"

    # エクスポート設定の全パターン (10パターン)
    export_configs = [
        {"format": "onnx",   "suffix": "onnx_fp32"},
        {"format": "onnx",   "suffix": "onnx_fp16",   "half": True},
        {"format": "tflite", "suffix": "tflite_fp32"},
        {"format": "tflite", "suffix": "tflite_fp16", "half": True},
        {"format": "tflite", "suffix": "tflite_int8", "int8": True, "data": dataset_yaml},
        {"format": "mnn",    "suffix": "mnn_fp32"},
        {"format": "mnn",    "suffix": "mnn_fp16",    "half": True},
        {"format": "mnn",    "suffix": "mnn_int8",    "int8": True, "data": dataset_yaml},
    ]

    total_exports = len(base_models) * len(image_sizes) * len(export_configs)
    current_count = 0

    print(f"=== 全{total_exports}パターンのエクスポートを開始します ===")

    for base_model_name in base_models:
        original_pt_path = models_dir / base_model_name

        # ベースモデルが存在しない場合は自動ダウンロードさせるため、一度ロードしておく
        if not original_pt_path.exists():
            print(f"{base_model_name} をダウンロード/準備中...")
            YOLO(str(original_pt_path))

        model_prefix = original_pt_path.stem # 'yolo26n' や 'yolo26s'

        for imgsz in image_sizes:
            for config in export_configs:
                current_count += 1

                # 妥当な名前の生成 (例: yolo26n_size320_ncnn_fp16)
                target_name = f"{model_prefix}_size{imgsz}_{config['suffix']}"
                temp_pt_path = output_dir / f"{target_name}.pt"

                print(f"\n[{current_count}/{total_exports}] 作成中: {target_name} ...")

                try:
                    # 1. 元の.ptファイルを、目的の名前でコピーする
                    shutil.copy(original_pt_path, temp_pt_path)

                    # 2. コピーしたモデルをロード
                    model = YOLO(str(temp_pt_path))

                    # 3. 引数を組み立ててエクスポート
                    # config辞書から format や half などの値を取り出す
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
                    # 4. 用済みになった一時的な .pt ファイルを削除（フォルダ内を綺麗に保つ）
                    if temp_pt_path.exists():
                        temp_pt_path.unlink()

    print("\n=== すべてのエクスポート処理が完了しました！ ===")
    print(f"出力先: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
