import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from main import CLASS_ID, CONFIDENCE, MODEL_PATH, detect_cat


def main():
    # データセットのパス設定
    dataset_dir = Path("dataset")
    with_cat_dir = dataset_dir / "with_cat"  # 猫がいる画像（正例）
    without_cat_dir = dataset_dir / "without_cat"  # 猫がいない画像（負例）

    # フォルダの存在確認
    if not with_cat_dir.exists() or not without_cat_dir.exists():
        print("エラー: データセットのディレクトリが見つかりません。")
        print("以下の構成でフォルダを作成し、画像を配置してください:")
        print(f"  - {with_cat_dir}/")
        print(f"  - {without_cat_dir}/")
        return

    # 対応する画像拡張子
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

    # 混同行列（Confusion Matrix）用のカウンター
    TP = 0  # True Positive:  猫がいる画像を「いる」と当てた
    FN = 0  # False Negative: 猫がいる画像を「いない」と間違えた
    TN = 0  # True Negative:  猫がいない画像を「いない」と当てた
    FP = 0  # False Positive: 猫がいない画像を「いる」と間違えた

    total_inference_time = 0.0
    inference_count = 0

    print("モデルのロード中...")
    try:
        model = YOLO(MODEL_PATH, task="detect")
    except Exception as e:
        print(f"モデルのロードに失敗しました: {e}")
        return

    print("\n=== 猫がいる画像 (with_cat) の検証開始 ===")
    for file_path in with_cat_dir.iterdir():
        if file_path.suffix.lower() in valid_extensions:
            frame = cv2.imread(str(file_path))
            if frame is None:
                continue

            # 推論実行（main.py の関数を利用）
            start_time = time.time()
            detected, conf, _ = detect_cat(
                model, frame, CLASS_ID, CONFIDENCE, imgsz=320
            )
            total_inference_time += time.time() - start_time
            inference_count += 1

            if detected:
                TP += 1
            else:
                FN += 1
                print(f"[FN] 見逃し: {file_path.name}")

    print("\n=== 猫がいない画像 (without_cat) の検証開始 ===")
    for file_path in without_cat_dir.iterdir():
        if file_path.suffix.lower() in valid_extensions:
            frame = cv2.imread(str(file_path))
            if frame is None:
                continue

            # 推論実行
            start_time = time.time()
            detected, conf, _ = detect_cat(
                model, frame, CLASS_ID, CONFIDENCE, imgsz=320
            )
            total_inference_time += time.time() - start_time
            inference_count += 1

            if detected:
                FP += 1
                print(f"[FP] 誤検知: {file_path.name} (信頼度: {conf:.2f})")
            else:
                TN += 1

    # --- 評価指標の計算 ---
    total_images = TP + TN + FP + FN

    if total_images == 0:
        print("\n評価対象の画像が見つかりませんでした。")
        return

    # 正解率 (Accuracy): 全体のうち、正しく予測できた割合
    accuracy = (TP + TN) / total_images if total_images > 0 else 0.0

    # 適合率 (Precision): 「猫がいる」と予測した中で、実際に猫がいた割合 (FPが少ないほど高い)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    # 再現率 (Recall): 実際に「猫がいる」画像の中で、正しく検出できた割合 (FNが少ないほど高い)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # F1スコア: PrecisionとRecallの調和平均 (両方のバランスを見る)
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # 結果の出力
    print("\n==================================")
    print("           検証結果レポート           ")
    print("==================================")
    print(f"Total Images: {total_images}")
    print("----------------------------------")
    print("[Confusion Matrix / 混同行列]")
    print(f"  TP (正解 - 検出成功) : {TP}")
    print(f"  TN (正解 - 無視成功) : {TN}")
    print(f"  FP (誤検知 - 誤作動) : {FP}")
    print(f"  FN (見逃し - 未検出) : {FN}")
    print("----------------------------------")
    print("[Metrics / 評価指標]")
    print(f"  Accuracy  (正解率) : {accuracy:.2%}")
    print(f"  Precision (適合率) : {precision:.2%}")
    print(f"  Recall    (再現率) : {recall:.2%}")
    print(f"  F1-Score (F1値)   : {f1_score:.2%}")

    avg_inference_time = (
        (total_inference_time / inference_count * 1000) if inference_count > 0 else 0
    )
    print("----------------------------------")
    print("[Performance / パフォーマンス]")
    print(f"  Avg Inference Time : {avg_inference_time:.1f} ms / image")
    print(
        f"  Total Inference    : {total_inference_time:.2f} sec used for {inference_count} images"
    )
    print("==================================")


if __name__ == "__main__":
    main()
