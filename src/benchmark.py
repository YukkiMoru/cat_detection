import csv
import gc
import subprocess
import sys
from pathlib import Path

import numpy as np

from inference import CatDetector

# precision.py に前回作成した evaluate_with_boxes がある想定
from precision import evaluate_with_boxes

N_RUNS = 5

# CSVの出力ヘッダー
FIELDNAMES = [
    "Model_Name",
    "Format",
    "Size",
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "Avg_Time(ms)",
    "FPS",
]


def main():
    print("=== YOLO モデル 一括最適化 & IoUベンチマーク ===")

    # 1. 準備
    current_dir = Path(__file__).parent
    tuning_script = (
        current_dir / "tune_v2.py"
    )  # 前回の最適化スクリプト名に合わせてください

    models_dir = Path("models")
    best_params_dir = Path("best_params")
    best_params_dir.mkdir(parents=True, exist_ok=True)

    # dataset_boxed を使用
    dataset_dir = Path("dataset_boxed/val")
    if not dataset_dir.exists():
        print(f"❌ エラー: ディレクトリが見つかりません: {dataset_dir}")
        return

    csv_path = Path("benchmark_results.csv")
    error_log_path = Path("error_models.txt")

    # 2. 進捗とエラー履歴の読み込み
    results = []
    evaluated_models = set()
    error_models = set()

    if csv_path.exists():
        try:
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    results.append(row)
                    evaluated_models.add(row["Model_Name"])
            print(
                f"🔄 既存の進行状況を読み込みました。検証済み: {len(evaluated_models)}件"
            )
        except Exception as e:
            print(f"⚠️ CSVの読み込み失敗: {e}")

    if error_log_path.exists():
        error_models = {
            line.strip()
            for line in error_log_path.read_text().splitlines()
            if line.strip()
        }
        print(f"🚫 スキップ対象(エラー履歴): {len(error_models)}件")

    # 3. モデル探索
    model_files = list(models_dir.rglob("*.onnx")) + list(models_dir.rglob("*.mnn"))
    model_files += [p for p in models_dir.rglob("*openvino_model") if p.is_dir()]
    model_files = [f for f in model_files if ".cache" not in str(f)]

    print(f"📦 合計 {len(model_files)} 個のモデルをスキャンします。")

    for i, model_path in enumerate(model_files, 1):
        m_name = model_path.stem
        if m_name in evaluated_models:
            print(f"\n--- [{i}/{len(model_files)}] {m_name} (⏭️ スキップ: 検証済み) ---")
            continue
        if m_name in error_models:
            print(
                f"\n--- [{i}/{len(model_files)}] {m_name} (🚫 スキップ: 過去のエラー) ---"
            )
            continue

        print(f"\n--- [{i}/{len(model_files)}] {model_path.name} ---")

        # 4. パラメータ最適化 (存在しない場合のみ実行)
        params_json = best_params_dir / f"{m_name}.json"
        if not params_json.exists():
            print("🔍 専用パラメータ未検出。最適化(Optuna)を開始...")
            try:
                subprocess.run(
                    [
                        sys.executable,
                        str(tuning_script),
                        "--model",
                        str(model_path),
                        "--dir",
                        str(dataset_dir),
                        "--trials",
                        "30",
                    ],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                print(f"⚠️ 最適化エラー: {e}")
                error_models.add(m_name)
                error_log_path.open("a").write(f"{m_name}\n")
                continue

        # 5. ベンチマーク実行
        try:
            detector = CatDetector(model_path=model_path)

            # 計測用変数の初期化
            metrics_list = []
            print(f"⏱️ ベンチマーク実行中 ({N_RUNS}回平均)...")

            for run_idx in range(N_RUNS):
                # IoUベースの評価関数を呼び出し
                m = evaluate_with_boxes(detector, dataset_dir, verbose=False)
                metrics_list.append(m)

            # 数値の平均化
            avg_acc = np.mean(
                [
                    (m["tp"] + m["tn"]) / (m["tp"] + m["tn"] + m["fp"] + m["fn"])
                    for m in metrics_list
                ]
            )
            avg_pre = np.mean([m["precision"] for m in metrics_list])
            avg_rec = np.mean([m["recall"] for m in metrics_list])
            avg_f1 = np.mean([m["f1"] for m in metrics_list])
            avg_ms = np.mean([m["avg_ms"] for m in metrics_list])
            final_fps = 1000.0 / avg_ms if avg_ms > 0 else 0

            # 結果格納
            row = {
                "Model_Name": m_name,
                "Format": "OPENVINO"
                if model_path.is_dir()
                else model_path.suffix[1:].upper(),
                "Size": detector.imgsz,
                "Accuracy": f"{avg_acc:.2%}",
                "Precision": f"{avg_pre:.2%}",
                "Recall": f"{avg_rec:.2%}",
                "F1_Score": f"{avg_f1:.2%}",
                "Avg_Time(ms)": f"{avg_ms:.1f}",
                "FPS": f"{final_fps:.1f}",
            }
            results.append(row)
            evaluated_models.add(m_name)

            # CSV保存（1モデルごとに更新してクラッシュ対策）
            results.sort(key=lambda x: float(x["F1_Score"].strip("%")), reverse=True)
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerows(results)

            print(f"✅ 完了: F1={avg_f1:.2%}, FPS={final_fps:.1f}")

        except Exception as e:
            print(f"❌ 評価エラー: {e}")
            error_log_path.open("a").write(f"{m_name}\n")
            error_models.add(m_name)

        finally:
            if "detector" in locals():
                del detector
            gc.collect()

    print("\n" + "=" * 45)
    print("🏆 ベンチマーク完了！全モデルの評価が終わりました。")
    print("=" * 45)


if __name__ == "__main__":
    main()
