#!/usr/bin/env python
# uv run clean_dataset.py

from pathlib import Path

# ==========================================
# ⚙️ 設定: 対象とするディレクトリ
# ==========================================
DATASET_DIRS = [Path("dataset_boxed/train"), Path("dataset_boxed/val")]
# 対応する画像拡張子
IMG_EXTS = {".jpg", ".jpeg", ".png"}
# ==========================================


def clean_dataset(base_dir: Path):
    img_dir = base_dir / "images"
    label_dir = base_dir / "labels"

    if not img_dir.exists() or not label_dir.exists():
        print(f"⚠️ スキップ: フォルダが見つかりません {base_dir}")
        return

    print(f"--- 📂 処理中: {base_dir} ---")

    # 1. 画像がないのにラベル（txt）があるものを削除
    for txt_path in label_dir.glob("*.txt"):
        # 画像ファイルがあるかチェック（複数の拡張子に対応）
        has_image = any(
            (img_dir / f"{txt_path.stem}{ext}").exists() for ext in IMG_EXTS
        )

        if not has_image:
            print(f"🗑️ 削除 (画像なしラベル): {txt_path.name}")
            txt_path.unlink()

    # 2. 画像があるのにラベル（txt）がないものを空ファイルで作成
    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() in IMG_EXTS:
            txt_path = label_dir / f"{img_path.stem}.txt"
            if not txt_path.exists():
                txt_path.touch()
                print(f"📄 作成 (空ラベル): {txt_path.name}")

    # 3. ラベルの内容を整形 (複数行ある場合、1行1ボックスを確実にする)
    for txt_path in label_dir.glob("*.txt"):
        if txt_path.stat().st_size == 0:
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # 空白行を除去し、各行の前後スペースを整える
        cleaned_lines = [line.strip() for line in lines if line.strip()]

        # 修正が必要な場合（不要な空白があった場合など）のみ上書き
        new_content = "\n".join(cleaned_lines)
        if len(cleaned_lines) > 0:
            new_content += "\n"  # 最後に改行を入れるのが一般的

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(new_content)


def main():
    for d in DATASET_DIRS:
        clean_dataset(d)
    print("\n✅ すべての処理が完了しました！")


if __name__ == "__main__":
    main()
