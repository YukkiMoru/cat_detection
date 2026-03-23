#!/usr/bin/env python
# uv run train/data_augment.py
# train/dataset_origin/background/1.png,2.png... & train/dataset_origin/cutted/1.png,2.png...
# -> train/dataset_augment

import glob
import os
import random

import cv2
import numpy as np

# --- 設定 ---
BASE_DIR = "train/dataset_origin"
BG_DIR = os.path.join(BASE_DIR, "background")
CAT_DIR = os.path.join(BASE_DIR, "cutted")
SAVE_DIR = "train/dataset_augment"

os.makedirs(os.path.join(SAVE_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "labels"), exist_ok=True)

NUM_GENERATE = 100


def augment_and_paste():
    bg_files = glob.glob(os.path.join(BG_DIR, "*.png"))
    cat_files = glob.glob(os.path.join(CAT_DIR, "*.png"))

    if not bg_files or not cat_files:
        print("エラー: 画像が見つかりません。")
        return

    print(f"{NUM_GENERATE}枚のコラ画像を生成中（下半分・大きめ）...")

    for i in range(NUM_GENERATE):
        bg = cv2.imread(random.choice(bg_files))
        cat = cv2.imread(random.choice(cat_files), cv2.IMREAD_UNCHANGED)

        if bg is None or cat is None:
            continue

        # --- 1. サイズを大きめに変更 (0.4〜0.7倍) ---
        scale = random.uniform(0.7, 1.4)
        cat = cv2.resize(cat, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # 反転と回転
        if random.random() > 0.5:
            cat = cv2.flip(cat, 1)
        h, w = cat.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-10, 10), 1)
        cat = cv2.warpAffine(
            cat,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        # 夜の暗さ調整
        brightness = random.uniform(0.8, 1.6)
        cat[:, :, :3] = (cat[:, :, :3].astype(np.float32) * brightness).astype(np.uint8)

        # --- 2. 出現位置を「半分より下」に制限 ---
        bg_h, bg_w = bg.shape[:2]
        cat_h, cat_w = cat.shape[:2]

        # x座標は左右端に寄りすぎないように調整
        x = random.randint(0, max(0, bg_w - cat_w))

        # y座標: 「画像の半分」から「下端」までの間で決める
        y_min = bg_h // 2
        y_max = max(y_min, bg_h - cat_h)
        y = random.randint(y_min, y_max)

        # 3. アルファブレンド
        alpha_cat = cat[:, :, 3] / 255.0
        alpha_bg = 1.0 - alpha_cat
        for c in range(3):
            bg[y : y + cat_h, x : x + cat_w, c] = (
                alpha_cat * cat[:, :, c]
                + alpha_bg * bg[y : y + cat_h, x : x + cat_w, c]
            )

        # 4. YOLO形式保存
        cx = (x + cat_w / 2) / bg_w
        cy = (y + cat_h / 2) / bg_h
        nw = cat_w / bg_w
        nh = cat_h / bg_h

        file_name = f"{i:04d}"
        cv2.imwrite(os.path.join(SAVE_DIR, "images", f"{file_name}.jpg"), bg)
        with open(os.path.join(SAVE_DIR, "labels", f"{file_name}.txt"), "w") as f:
            f.write(f"15 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    print(f"完了: {SAVE_DIR} に保存されました。")


if __name__ == "__main__":
    augment_and_paste()
