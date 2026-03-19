import argparse
import time
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from inference import CatDetector

# --- ヘルパー関数 ---


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    2つのボックス(x1, y1, x2, y2)のIoUを計算する
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def load_yolo_labels(path: Path) -> np.ndarray:
    """
    YOLO形式のtxtからボックスリスト(xyxy, 正規化)を読み込む
    """
    boxes = []
    if not path.exists():
        return np.array([])

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # class_id, cx, cy, w, h
            cx, cy, bw, bh = map(float, parts[1:5])
            x1, y1 = cx - bw / 2, cy - bh / 2
            x2, y2 = cx + bw / 2, cy + bh / 2
            boxes.append([x1, y1, x2, y2])

    return np.array(boxes)


def evaluate_with_boxes(
    detector: CatDetector,
    dataset_dir: Path,
    iou_threshold: float = 0.45,
    verbose: bool = False,
) -> Dict:
    """
    ボックスの重なり(IoU)を考慮して精度を評価する
    """
    img_dir = dataset_dir / "images"
    label_dir = dataset_dir / "labels"

    image_paths = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )

    stats = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    inference_times = []

    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        # 1. 推論
        t1 = time.perf_counter()
        detected, conf, results = detector.detect_cat(frame)
        inference_times.append(time.perf_counter() - t1)

        # 検出ボックスの取得 (xyxy, 正規化座標)
        pred_boxes = []
        if detected and results:
            # results[0].boxes.xyxyn は 0.0~1.0 の正規化座標
            pred_boxes = results[0].boxes.xyxyn.cpu().numpy()

        # 2. 正解(GT)ラベルの読み込み
        label_path = label_dir / f"{img_path.stem}.txt"
        gt_boxes = load_yolo_labels(label_path)

        # 3. 判定ロジック
        if len(gt_boxes) == 0:
            # 【背景画像の場合】
            if len(pred_boxes) == 0:
                stats["tn"] += 1  # 正解：何も出なかった
            else:
                stats["fp"] += len(pred_boxes)  # 誤検知：何か出た
                if verbose:
                    print(f"[FP] 背景なのに検知: {img_path.name}")
        else:
            # 【猫がいる画像の場合】
            matched_gt = set()
            for p_box in pred_boxes:
                best_iou = 0
                best_gt_idx = -1
                for i, g_box in enumerate(gt_boxes):
                    iou = calculate_iou(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i

                if best_iou >= iou_threshold:
                    if best_gt_idx not in matched_gt:
                        stats["tp"] += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        stats["fp"] += 1  # 同じGTに2つ以上の枠が出た場合はFP扱い
                else:
                    stats["fp"] += 1  # 枠はあるが、場所が全然違う

            # 見逃したGTの数
            fn_count = len(gt_boxes) - len(matched_gt)
            stats["fn"] += fn_count
            if verbose and fn_count > 0:
                print(f"[FN] 見逃し: {img_path.name}")

    # メトリクス計算
    tp, fp, fn, tn = stats["tp"], stats["fp"], stats["fn"], stats["tn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        **stats,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_ms": np.mean(inference_times) * 1000,
    }


# --- 実行メイン ---


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", type=str, default="dataset_boxed/val", help="評価するディレクトリ"
    )
    args = parser.parse_args()

    target_dir = Path(args.dir)
    detector = CatDetector()  # モデルとベストパラメータは自動ロードされる

    print(f"🚀 検証開始: {target_dir}")
    print(f"📦 使用モデル: {detector.model_path.name}")
    print(f"🛠️ 前処理設定: {detector.params}")
    print("-" * 30)

    results = evaluate_with_boxes(detector, target_dir, verbose=True)

    print("\n" + "=" * 40)
    print(f" 📊 評価レポート: {target_dir.name}")
    print("=" * 40)
    print(f" 🟢 TP (正解): {results['tp']:>4} |  🔴 FP (誤検知): {results['fp']:>4}")
    print(f" ⚪ TN (無視): {results['tn']:>4} |  🟡 FN (見逃し): {results['fn']:>4}")
    print("-" * 40)
    print(f" 🎯 Precision : {results['precision']:.2%}")
    print(f" 📢 Recall    : {results['recall']:.2%}")
    print(f" 🏆 F1-Score  : {results['f1']:.2%}")
    print(f" ⚡ Speed     : {results['avg_ms']:.1f} ms/image")
    print("=" * 40)


if __name__ == "__main__":
    main()
