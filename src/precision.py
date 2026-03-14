import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2

import config
from inference import CatDetector

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def load_image_paths(directory: Path) -> List[Path]:
    """ディレクトリから画像ファイルのパスだけを収集する（フレームは保持しない）。"""

    paths: List[Path] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        paths.append(path)
    return paths


def evaluate(
    detector: CatDetector,
    with_cat_images: List[Path],
    without_cat_images: List[Path],
    params: Dict | None = None,
    verbose: bool = False,
    warmup: bool = True,
) -> Dict:
    """
    猫検出の評価を行い、混同行列と各種メトリクスを返す。

    verbose=True の場合、FN/FP の詳細を標準出力に表示する。
    """
    tp = fn = tn = fp = 0
    io_time_sec = 0.0
    inference_time_sec = 0.0
    original_params = detector.params.copy()
    if params is not None:
        detector.set_params(params)

    # --- ウォームアップ ---
    # 初回のモデル初期化/キャッシュ生成が計測に混ざらないよう、1枚だけ推論しておく
    if warmup:
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

    if verbose:
        print("\n=== 猫がいる画像 (with_cat) の検証開始 ===")

    for path in with_cat_images:
        name = path.name
        t0 = time.perf_counter()
        frame = cv2.imread(str(path))
        io_time_sec += time.perf_counter() - t0
        if frame is None:
            continue

        t1 = time.perf_counter()
        detected, _, _ = detector.detect_cat(frame)
        inference_time_sec += time.perf_counter() - t1
        if detected:
            tp += 1
        else:
            fn += 1
            if verbose:
                print(f"[FN] 見逃し: {name}")

    if verbose:
        print("\n=== 猫がいない画像 (without_cat) の検証開始 ===")

    for path in without_cat_images:
        name = path.name
        t0 = time.perf_counter()
        frame = cv2.imread(str(path))
        io_time_sec += time.perf_counter() - t0
        if frame is None:
            continue

        t1 = time.perf_counter()
        detected, conf, _ = detector.detect_cat(frame)
        inference_time_sec += time.perf_counter() - t1
        if detected:
            fp += 1
            if verbose:
                print(f"[FP] 誤検知: {name} (信頼度: {conf:.2f})")
        else:
            tn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )

    detector.set_params(original_params)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "io_time_sec": io_time_sec,
        "inference_time_sec": inference_time_sec,
    }


def main():
    parser = argparse.ArgumentParser(description="猫検出モデルの精度を評価する")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="評価するモデルのパス (指定しない場合は config.MODEL_PATH)",
    )
    args = parser.parse_args()

    dataset_dir = Path("dataset")
    with_cat_dir = dataset_dir / "with_cat"
    without_cat_dir = dataset_dir / "without_cat"

    if not with_cat_dir.exists() or not without_cat_dir.exists():
        print("エラー: データセットのディレクトリが見つかりません。")
        print("以下の構成でフォルダを作成し、画像を配置してください:")
        print(f"  - {with_cat_dir}/")
        print(f"  - {without_cat_dir}/")
        return

    # --- 修正ポイント: モデル名から専用のチューニング済みパラメータを自動で探す ---
    target_model_path = args.model if args.model else Path(config.MODEL_PATH)
    model_stem = target_model_path.stem
    params_path = Path("best_params") / f"{model_stem}.json"

    print(f"モデルのロード中... : {target_model_path.name}")
    if params_path.exists():
        print(f"専用パラメータを適用します: {params_path.name}")
    else:
        print("専用パラメータが見つからないため、デフォルト設定で評価します。")
        params_path = None  # Noneを渡すと inference.py 側でデフォルトが使われる

    try:
        # モデルパスとパラメータパスを両方渡して初期化
        detector = CatDetector(model_path=target_model_path, params_path=params_path)
    except Exception as e:
        print(f"モデルのロードに失敗しました: {e}")
        return

    with_cat_images = load_image_paths(with_cat_dir)
    without_cat_images = load_image_paths(without_cat_dir)

    if not with_cat_images and not without_cat_images:
        print("\n評価対象の画像が見つかりませんでした。")
        return

    metrics = evaluate(
        detector,
        with_cat_images,
        without_cat_images,
        verbose=True,
    )
    total_images = metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"]

    avg_inference_time = (
        (metrics["inference_time_sec"] / total_images * 1000) if total_images > 0 else 0
    )

    print("\n==================================")
    print(f"    検証結果レポート ({target_model_path.name})")
    print("==================================")
    print(f"Total Images: {total_images}")
    print("----------------------------------")
    print("[Confusion Matrix / 混同行列]")
    print(f"  TP (正解 - 検出成功) : {metrics['tp']}")
    print(f"  TN (正解 - 無視成功) : {metrics['tn']}")
    print(f"  FP (誤検知 - 誤作動) : {metrics['fp']}")
    print(f"  FN (見逃し - 未検出) : {metrics['fn']}")
    print("----------------------------------")
    print("[Metrics / 評価指標]")
    print(f"  Accuracy  (正解率) : {metrics['accuracy']:.2%}")
    print(f"  Precision (適合率) : {metrics['precision']:.2%}")
    print(f"  Recall    (再現率) : {metrics['recall']:.2%}")
    print(f"  F1-Score (F1値)   : {metrics['f1']:.2%}")
    print("----------------------------------")
    print("[Performance / パフォーマンス]")
    print(f"  Avg Inference Time : {avg_inference_time:.1f} ms / image")
    print(
        f"  Total Inference    : {metrics['inference_time_sec']:.2f} sec used for {total_images} images"
    )
    print(f"  Total I/O Read      : {metrics['io_time_sec']:.2f} sec")
    print("==================================")


if __name__ == "__main__":
    main()
