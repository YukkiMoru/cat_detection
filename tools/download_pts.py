import shutil
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from ultralytics import YOLO

# --- 設定 ---
MODELS_DIR = Path("models")
BASE_MODELS = ["yolo26n.pt", "yolo26s.pt"]
IMAGE_SIZES = [640, 416, 320]
DATASET_YAML = "coco8.yaml"
# CPUの芯数に合わせて調整（例: 4〜8程度が安全。Xeonならもう少し増やせますがメモリと相談）
MAX_WORKERS = 8

EXPORT_CONFIGS = [
    {"format": "onnx", "suffix": "onnx_fp32"},
    {"format": "onnx", "suffix": "onnx_fp16", "half": True},
    {"format": "tflite", "suffix": "tflite_fp32"},
    {"format": "tflite", "suffix": "tflite_fp16", "half": True},
    {"format": "tflite", "suffix": "tflite_int8", "int8": True, "data": DATASET_YAML},
    {"format": "mnn", "suffix": "mnn_fp32"},
    {"format": "mnn", "suffix": "mnn_fp16", "half": True},
    {"format": "mnn", "suffix": "mnn_int8", "int8": True, "data": DATASET_YAML},
]

def export_single_pattern(args):
    """
    1つのパターンをエクスポートする関数（この関数が並列で呼ばれる）
    """
    base_model_path, imgsz, config, target_dir = args
    model_prefix = base_model_path.stem
    target_name = f"{model_prefix}_size{imgsz}_{config['suffix']}"
    temp_pt_path = target_dir / f"{target_name}.pt"
    
    # 既に出力があるか確認
    existing_outputs = list(target_dir.glob(f"{target_name}.*"))
    if existing_outputs:
        return f"[スキップ] {target_name}"

    try:
        # 1. ptファイルのコピー（並列実行時の競合を避けるため個別にコピー）
        shutil.copy(base_model_path, temp_pt_path)
        
        # 2. ロード
        model = YOLO(str(temp_pt_path))
        
        # 3. エクスポート実行
        model.export(
            format=config["format"],
            imgsz=imgsz,
            half=config.get("half", False),
            int8=config.get("int8", False),
            data=config.get("data"),
            opset=17, # 互換性のために追加
            verbose=False # ログが混ざるのを防ぐ
        )
        return f"[成功] {target_name}"
    except Exception as e:
        return f"[エラー] {target_name}: {e}"
    finally:
        if temp_pt_path.exists():
            temp_pt_path.unlink()

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 事前に全ベースモデルをダウンロード
    for name in BASE_MODELS:
        p = MODELS_DIR / name
        if not p.exists():
            YOLO(name) # 自動ダウンロード
            shutil.move(name, p)

    # タスクリストの作成
    tasks = []
    for model_name in BASE_MODELS:
        base_path = MODELS_DIR / model_name
        target_dir = MODELS_DIR / base_path.stem
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for imgsz in IMAGE_SIZES:
            for config in EXPORT_CONFIGS:
                tasks.append((base_path, imgsz, config, target_dir))

    print(f"=== {len(tasks)} パターンの並列エクスポートを開始 (同時実行数: {MAX_WORKERS}) ===")

    # 並列実行
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(export_single_pattern, tasks))

    for res in results:
        print(res)

    print("\n=== すべての処理が完了しました ===")

if __name__ == "__main__":
    main()