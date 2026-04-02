#!/usr/bin/env python
import shutil
from pathlib import Path

# ==========================================
# ⚙️ 設定
# ==========================================
# 元データのパス
SRC_IMG_DIR = Path("dataset_boxed2/original/images")
SRC_LBL_DIR = Path("dataset_boxed2/original/labels")

# 振り分け先のルート
EXPORT_ROOT = Path("dataset_boxed3/original")
CAT_DIR = EXPORT_ROOT / "cat"
NOT_DIR = EXPORT_ROOT / "not"
# ==========================================


def setup_dirs(base_path):
    (base_path / "images").mkdir(parents=True, exist_ok=True)
    (base_path / "labels").mkdir(parents=True, exist_ok=True)


def main():
    # フォルダ準備
    setup_dirs(CAT_DIR)
    setup_dirs(NOT_DIR)

    # 画像拡張子のリスト
    exts = ("*.jpg", "*.jpeg", "*.png", "*.PNG", "*.JPG")
    image_paths = []
    for ext in exts:
        image_paths.extend(list(SRC_IMG_DIR.glob(ext)))

    if not image_paths:
        print(f"❌ 画像が見つかりませんでした: {SRC_IMG_DIR}")
        return

    print(f"探査中... {len(image_paths)} 枚の画像を処理します。")

    count_cat = 0
    count_not = 0

    for img_path in image_paths:
        # 対応するラベルファイルのパス
        label_path = SRC_LBL_DIR / f"{img_path.stem}.txt"

        # ボックス（ラベル）があるか判定
        # 1. ファイルが存在する
        # 2. 中身が空ではない（1文字以上ある）
        has_box = False
        if label_path.exists():
            with open(label_path, "r") as f:
                content = f.read().strip()
                if content:
                    has_box = True

        # 振り分け先の決定
        target_folder = CAT_DIR if has_box else NOT_DIR

        # 画像のコピー
        shutil.copy2(img_path, target_folder / "images" / img_path.name)

        # ラベルのコピー（ボックスがある場合のみ）
        if has_box:
            shutil.copy2(label_path, target_folder / "labels" / label_path.name)
            count_cat += 1
        else:
            count_not += 1

    print("-" * 30)
    print("✅ 処理完了！")
    print(f"📂 {CAT_DIR.name} (ボックスあり): {count_cat} 枚")
    print(f"📂 {NOT_DIR.name} (ボックスなし): {count_not} 枚")
    print("-" * 30)


if __name__ == "__main__":
    main()
