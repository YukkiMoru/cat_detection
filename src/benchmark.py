import csv
import subprocess
import sys  # 実行中のPythonパスを取得するために追加
import time
from pathlib import Path

from inference import CatDetector
from precision import evaluate, load_images


def main():
    print("=== YOLO モデル 一括最適化 & ベンチマーク ===")

    # 1. 準備
    current_dir = Path(__file__).parent  # このファイルがあるディレクトリ (src)
    tuning_script = current_dir / "tuning.py"

    models_dir = Path("models")
    best_params_dir = Path("best_params")
    best_params_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path("dataset")
    with_cat_dir = dataset_dir / "with_cat"
    without_cat_dir = dataset_dir / "without_cat"

    print("画像を読み込んでいます...")
    with_cat_images = load_images(with_cat_dir)
    without_cat_images = load_images(without_cat_dir)
    total_images = len(with_cat_images) + len(without_cat_images)

    # 2. モデルの自動探索
    model_files = list(models_dir.rglob("*.onnx")) + list(models_dir.rglob("*.mnn"))
    print(f"合計 {len(model_files)} 個のモデルをチェックします。")

    results = []

    for i, model_path in enumerate(model_files, 1):
        print(f"\n--- [{i}/{len(model_files)}] {model_path.name} ---")

        params_json = best_params_dir / f"{model_path.stem}.json"
        if not params_json.exists():
            print("🔍 パラメータが見つかりません。最適化を開始します...")
            try:
                # sys.executable を使うことで、仮想環境(venv)のPythonを確実に引き継ぎます
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

        # 3. 最適化された状態で評価
        try:
            detector = CatDetector(model_path=model_path)

            start_time = time.time()
            metrics = evaluate(
                detector, with_cat_images, without_cat_images, verbose=False
            )
            total_time = time.time() - start_time

            avg_ms = (total_time / total_images * 1000) if total_images else 0
            fps = (1000 / avg_ms) if avg_ms else 0

            results.append(
                {
                    "Model_Name": model_path.stem,
                    "Format": model_path.suffix.replace(".", "").upper(),
                    "Size": detector.imgsz,
                    "Accuracy": f"{metrics['accuracy']:.2%}",
                    "Precision": f"{metrics['precision']:.2%}",
                    "Recall": f"{metrics['recall']:.2%}",
                    "F1_Score": f"{metrics['f1']:.2%}",
                    "Avg_Time(ms)": f"{avg_ms:.1f}",
                    "FPS": f"{fps:.1f}",
                }
            )
            print(f"📈 評価結果: F1={metrics['f1']:.2%}, FPS={fps:.1f}")

        except Exception as e:
            print(f"❌ エラー発生: {e}")

    # 4. ソートと保存
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
    print("             🏆 最終ベンチマーク結果 🏆")
    print("=" * 45)
    for i, res in enumerate(results[:5], 1):
        print(f"{i}位: {res['Model_Name']}")
        print(f"     F1: {res['F1_Score']} | FPS: {res['FPS']} | Size: {res['Size']}")
    print("=" * 45)


if __name__ == "__main__":
    main()
