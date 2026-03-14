import csv
import subprocess
import sys
import time
from pathlib import Path

import cv2

from inference import CatDetector
from precision import evaluate, load_image_paths

N_RUNS = 5


def main():
    print("=== YOLO モデル 一括最適化 & ベンチマーク ===")

    # 1. 準備 (変更なし)
    current_dir = Path(__file__).parent
    tuning_script = current_dir / "tuning.py"

    models_dir = Path("models")
    best_params_dir = Path("best_params")
    best_params_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path("dataset")
    with_cat_dir = dataset_dir / "with_cat"
    without_cat_dir = dataset_dir / "without_cat"

    print("画像を読み込んでいます...")
    with_cat_images = load_image_paths(with_cat_dir)
    without_cat_images = load_image_paths(without_cat_dir)
    total_images = len(with_cat_images) + len(without_cat_images)

    # 2. モデルの自動探索 (変更なし)
    model_files = list(models_dir.rglob("*.onnx")) + list(models_dir.rglob("*.mnn"))
    # models/.cache を除外
    model_files = [f for f in model_files if ".cache" not in str(f)]
    print(f"合計 {len(model_files)} 個のモデルをチェックします。")

    results = []

    for i, model_path in enumerate(model_files, 1):
        print(f"\n--- [{i}/{len(model_files)}] {model_path.name} ---")

        params_json = best_params_dir / f"{model_path.stem}.json"
        if not params_json.exists():
            print("🔍 パラメータが見つかりません。最適化を開始します...")
            try:
                subprocess.run(
                    [
                        sys.executable,
                        str(tuning_script),
                        "--model",
                        str(model_path),
                        "--trials",
                        "30",
                    ],
                    check=True,
                )
                print(f"✅ 最適化完了: {params_json.name}")
            except subprocess.CalledProcessError as e:
                print(f"⚠️ チューニング中にエラーが発生しました（スキップします）: {e}")
                continue
        else:
            print("✨ チューニング済みパラメータを適用します。")

        # 3. 最適化された状態で評価（★ ここをN回平均に修正）
        try:
            detector = CatDetector(model_path=model_path)

            # ウォームアップ（初回ロード/キャッシュ生成を計測に含めない）
            warmup_path = (with_cat_images[0] if with_cat_images else None) or (
                without_cat_images[0] if without_cat_images else None
            )
            if warmup_path is not None:
                warmup_frame = cv2.imread(str(warmup_path))
                if warmup_frame is not None:
                    try:
                        detector.detect_cat(warmup_frame)
                    except Exception:
                        pass

            total_accuracy = 0.0
            total_precision = 0.0
            total_recall = 0.0
            total_f1 = 0.0
            total_avg_ms = 0.0
            total_avg_io_ms = 0.0

            print(f"ベンチマークを {N_RUNS} 回実行して平均を計測中...")
            for _ in range(N_RUNS):
                metrics = evaluate(
                    detector,
                    with_cat_images,
                    without_cat_images,
                    verbose=False,
                    warmup=False,
                )

                # 1周あたりの1枚推論時間（前処理+推論のみ）
                avg_ms_per_run = (
                    (metrics["inference_time_sec"] / total_images * 1000)
                    if total_images
                    else 0
                )
                avg_io_ms_per_run = (
                    (metrics["io_time_sec"] / total_images * 1000) if total_images else 0
                )

                total_accuracy += metrics["accuracy"]
                total_precision += metrics["precision"]
                total_recall += metrics["recall"]
                total_f1 += metrics["f1"]
                total_avg_ms += avg_ms_per_run
                total_avg_io_ms += avg_io_ms_per_run

            # 平均値の計算
            avg_accuracy = total_accuracy / N_RUNS
            avg_precision = total_precision / N_RUNS
            avg_recall = total_recall / N_RUNS
            avg_f1 = total_f1 / N_RUNS
            final_avg_ms = total_avg_ms / N_RUNS
            final_avg_io_ms = total_avg_io_ms / N_RUNS
            final_fps = (1000 / final_avg_ms) if final_avg_ms else 0

            results.append(
                {
                    "Model_Name": model_path.stem,
                    "Format": model_path.suffix.replace(".", "").upper(),
                    "Size": detector.imgsz,
                    "Accuracy": f"{avg_accuracy:.2%}",
                    "Precision": f"{avg_precision:.2%}",
                    "Recall": f"{avg_recall:.2%}",
                    "F1_Score": f"{avg_f1:.2%}",
                    "Avg_Time(ms)": f"{final_avg_ms:.1f}",
                    "Avg_IO(ms)": f"{final_avg_io_ms:.1f}",
                    "FPS": f"{final_fps:.1f}",
                }
            )
            print(f"📈 評価結果({N_RUNS}回平均): F1={avg_f1:.2%}, FPS={final_fps:.1f}")

        except Exception as e:
            print(f"❌ エラー発生: {e}")

    # 4. ソートと保存 (変更なし)
    if not results:
        print("有効な結果が得られませんでした。")
        return

    results.sort(
        key=lambda x: (float(x["F1_Score"].strip("%")), float(x["FPS"])), reverse=True
    )

    csv_path = Path("benchmark_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 45)
    print("            🏆 最終ベンチマーク結果 🏆")
    print("=" * 45)
    for i, res in enumerate(results[:5], 1):
        print(f"{i}位: {res['Model_Name']}")
        print(f"     F1: {res['F1_Score']} | FPS: {res['FPS']} | Size: {res['Size']}")
    print("=" * 45)


if __name__ == "__main__":
    main()
