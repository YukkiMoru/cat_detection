#!/usr/bin/env python
# uv run view_dataset.py

from pathlib import Path

import cv2

# ==========================================
# ⚙️ 設定
# ==========================================
# 参照したいデータセットのディレクトリをリストで指定します
# DATASET_DIRS = [Path("train/dataset_augment")]
DATASET_DIRS = [Path("dataset_boxed/val")]
# DATASET_DIRS = [Path("dataset_captured/1/train")]
# DATASET_DIRS = [Path("dataset_boxed2/val")]
# ==========================================


def get_all_image_paths(dataset_dirs):
    """指定された複数のディレクトリから画像パスを全て集める"""
    image_paths = []
    for d_dir in dataset_dirs:
        img_dir = d_dir / "images"
        if not img_dir.exists():
            print(f"⚠️ フォルダが見つかりません: {img_dir}")
            continue

        # 画像ファイルを検索 (.jpg, .png など)
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_paths.extend(list(img_dir.glob(ext)))

    # 名前順にソートしておく
    image_paths.sort()
    return image_paths


def draw_bboxes(image, label_path):
    """YOLO形式のテキストファイルから枠を描画する"""
    if not label_path.exists():
        return image

    h, w = image.shape[:2]

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        class_id = 15
        cx = float(parts[1])
        cy = float(parts[2])
        bw = float(parts[3])
        bh = float(parts[4])

        # YOLOの正規化座標(0.0~1.0)をピクセル座標に変換
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        # 緑色の枠とクラスIDを描画
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 背景付きテキストで文字を見やすくする
        label = f"ID: {class_id}"
        cv2.putText(
            image,
            label,
            (x1, max(y1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return image


def main():
    print("📂 画像を検索しています...")
    image_paths = get_all_image_paths(DATASET_DIRS)

    if not image_paths:
        print("❌ 画像が見つかりませんでした。パスを確認してください。")
        return

    total_images = len(image_paths)
    current_idx = 0

    print(f"✅ 合計 {total_images} 枚の画像が見つかりました！")
    print("🎮 操作方法: [T] 次の画像 / [R] 前の画像 / [Q] 終了")

    cv2.namedWindow("Dataset Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dataset Viewer", 960, 540)

    while True:
        img_path = image_paths[current_idx]
        # ラベルのパスを作成 (imagesフォルダをlabelsに、拡張子を.txtに変更)
        dataset_name = img_path.parent.parent.name  # dataset_yolo 等
        label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"⚠️ 読み込みエラー: {img_path.name}")
            current_idx = (current_idx + 1) % total_images
            continue

        # 1. バウンディングボックスを描画
        image = draw_bboxes(image, label_path)

        # 2. 画面左上に情報を表示 (現在のデータセット名とファイル名)
        info_text = (
            f"[{current_idx + 1}/{total_images}] {dataset_name} | {img_path.name}"
        )
        # 文字が見やすいように黒い枠付きで描画
        cv2.putText(
            image, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4
        )
        cv2.putText(
            image,
            info_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        # 3. 画面サイズに合わせて縮小して表示
        display_img = cv2.resize(image, (960, 540))
        cv2.imshow("Dataset Viewer", display_img)

        # 4. キー入力を待機 (0なのでキーが押されるまでストップ)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):  # Qキー: 終了
            break
        elif key == ord("r"):  # Rキー: 前に戻る (Reverse)
            current_idx = (current_idx - 1) % total_images
        elif key == ord("t"):  # Tキー: 次に進む
            current_idx = (current_idx + 1) % total_images
        elif key == ord("d"):  # Dキー: ラベルを削除して「背景（ラベルなし）」にする
            if label_path.exists():
                label_path.unlink()  # .txtファイルを物理的に削除
                print(f"ラベルを削除しました（背景に設定）: {label_path.name}")
            else:
                print(f"すでにラベルがありません: {label_path.name}")

            # 削除後、画面を更新して枠が消えたことを確認させるために再描画
            continue

    cv2.destroyAllWindows()
    print("🛑 終了しました")


if __name__ == "__main__":
    main()
