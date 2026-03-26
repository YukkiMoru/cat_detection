#!/usr/bin/env python
# uv run train/data_augment.py
# train/dataset_origin/background/1.png,2.png... & train/dataset_origin/cutted/1.png,2.png...
# -> train/dataset_augment/train/

import glob
import os
import random

import cv2
import numpy as np

# --- 設定 ---
BASE_DIR = "train/dataset_origin"
BG_DIR = os.path.join(BASE_DIR, "background")
CAT_DIR = os.path.join(BASE_DIR, "cutted")
SAVE_DIR = "train/dataset_augment/train"

os.makedirs(os.path.join(SAVE_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "labels"), exist_ok=True)

NUM_GENERATE = 1000
NEGATIVE_RATIO = 0.1  # 10%の確率で「ターゲットがいない画像（背景のみ）」を生成
CLASS_ID = 15  # 対象のクラスID


def rotate_background_safe(image, angle):
    """
    背景を少し大きくリサイズしてから回転・切り抜きすることで、
    四隅に不自然な反射模様や黒い隙間ができないようにする。
    """
    h, w = image.shape[:2]
    # 5度の回転に耐えられるよう1.1倍に拡大
    scale_factor = 1.1
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 回転
    M = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), angle, 1.0)
    rotated = cv2.warpAffine(image_resized, M, (new_w, new_h), flags=cv2.INTER_LINEAR)

    # 元のサイズに中央を切り抜き
    start_x = (new_w - w) // 2
    start_y = (new_h - h) // 2
    return rotated[start_y : start_y + h, start_x : start_x + w]


def augment_and_paste():
    bg_files = glob.glob(os.path.join(BG_DIR, "*.png"))
    cat_files = glob.glob(os.path.join(CAT_DIR, "*.png"))

    if not bg_files or not cat_files:
        print("エラー: 画像が見つかりません。")
        return

    print(f"{NUM_GENERATE}枚の生成を開始します（ネガティブサンプル込）...")

    for i in range(NUM_GENERATE):
        bg = cv2.imread(random.choice(bg_files))
        if bg is None:
            continue

        # --- 1. 背景の加工（安全な回転） ---
        bg_angle = random.uniform(-5, 5)
        bg = rotate_background_safe(bg, bg_angle)
        bg_h, bg_w = bg.shape[:2]

        file_name = f"{i:04d}"
        is_negative = random.random() < NEGATIVE_RATIO

        if is_negative:
            # ネガティブサンプルの場合、猫を貼らずに保存（ラベルは空）
            cv2.imwrite(os.path.join(SAVE_DIR, "images", f"{file_name}.jpg"), bg)
            open(os.path.join(SAVE_DIR, "labels", f"{file_name}.txt"), "w").close()
            continue

        # --- 2. 前景（ターゲット）の加工 ---
        cat = cv2.imread(random.choice(cat_files), cv2.IMREAD_UNCHANGED)
        if cat is None:
            continue

        # サイズ変更
        scale = random.uniform(0.7, 1.4)
        cat = cv2.resize(cat, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # 反転
        if random.random() > 0.5:
            cat = cv2.flip(cat, 1)

        # 回転（ターゲット自身）
        h, w = cat.shape[:2]
        M_cat = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-10, 10), 1)
        cat = cv2.warpAffine(
            cat,
            M_cat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        # 明るさ調整
        brightness = random.uniform(0.8, 1.4)
        cat_rgb = cat[:, :, :3].astype(np.float32) * brightness
        cat[:, :, :3] = np.clip(cat_rgb, 0, 255).astype(np.uint8)

        # --- 3. 貼り付け位置の計算 ---
        cat_h, cat_w = cat.shape[:2]
        x = random.randint(0, max(0, bg_w - cat_w))
        y_min = bg_h // 3  # 下半分に寄りすぎないよう調整
        y_max = max(y_min, bg_h - cat_h)
        y = random.randint(y_min, y_max)

        # --- 4. アルファブレンド（境界の馴染ませ込） ---
        alpha_cat = cat[:, :, 3].astype(np.float32) / 255.0
        # 輪郭をわずかにぼかして背景との境界を自然にする
        alpha_cat = cv2.GaussianBlur(alpha_cat, (3, 3), 0)

        alpha_bg = 1.0 - alpha_cat

        # 貼り付け範囲のチェック
        if y + cat_h > bg_h or x + cat_w > bg_w:
            continue

        roi = bg[y : y + cat_h, x : x + cat_w]
        for c in range(3):
            bg[y : y + cat_h, x : x + cat_w, c] = (
                alpha_cat * cat[:, :, c] + alpha_bg * roi[:, :, c]
            )

        # --- 5. YOLO形式保存 ---
        cx = (x + cat_w / 2) / bg_w
        cy = (y + cat_h / 2) / bg_h
        nw = cat_w / bg_w
        nh = cat_h / bg_h

        cv2.imwrite(os.path.join(SAVE_DIR, "images", f"{file_name}.jpg"), bg)
        with open(os.path.join(SAVE_DIR, "labels", f"{file_name}.txt"), "w") as f:
            f.write(f"{CLASS_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    print(f"完了: {SAVE_DIR} に {NUM_GENERATE}枚の画像とラベルが生成されました。")


if __name__ == "__main__":
    augment_and_paste()
