#!/usr/bin/env python
# uv run train/data_augment.py
# train/dataset_origin/background/1.png,2.png... & train/dataset_origin/cutted/1.png,2.png...
# -> train/dataset_augment
#!/usr/bin/env python
import concurrent.futures
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

NUM_GENERATE = 1000
CLASS_ID = 15  # YOLOのクラスID


def rotate_and_crop(image, angle):
    """画像を見切れずに回転させ、不透明部分のギリギリでクロップする"""
    h, w = image.shape[:2]
    diagonal = int(np.ceil(np.sqrt(h**2 + w**2)))
    new_image = np.zeros((diagonal, diagonal, 4), dtype=np.uint8)

    x_offset = (diagonal - w) // 2
    y_offset = (diagonal - h) // 2
    new_image[y_offset : y_offset + h, x_offset : x_offset + w] = image

    M = cv2.getRotationMatrix2D((diagonal / 2, diagonal / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        new_image,
        M,
        (diagonal, diagonal),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    coords = cv2.findNonZero(rotated[:, :, 3])
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        return rotated[y : y + h, x : x + w]
    return rotated


def apply_street_light_effect(image):
    """【高速化】スライスによる街灯風の色味とコントラスト追加"""
    # RGBではなくBGRの順であることに注意
    b_factor = random.uniform(0.8, 1.0)
    g_factor = random.uniform(1.0, 1.2)
    r_factor = random.uniform(1.0, 1.3)

    # float32で計算してからクリップ (split / merge を使わない)
    bgr = image[:, :, :3].astype(np.float32)
    bgr[:, :, 0] *= b_factor
    bgr[:, :, 1] *= g_factor
    bgr[:, :, 2] *= r_factor
    np.clip(bgr, 0, 255, out=bgr)

    # コントラスト調整
    alpha = random.uniform(1.1, 1.4)
    beta = random.randint(-20, 0)
    image[:, :, :3] = cv2.convertScaleAbs(bgr, alpha=alpha, beta=beta)

    return image


def add_iso_noise(image, intensity_range=(0.01, 0.04)):
    """【高速化】OpenCV関数を使用した高速なノイズ追加"""
    intensity = random.uniform(*intensity_range)
    sigma = intensity * 255

    # np.random.normal は遅いため、cv2.randn を使用
    noise = np.zeros(image.shape, dtype=np.int16)
    cv2.randn(noise, 0, sigma)

    noisy_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy_image


def generate_single_sample(idx, bg_files, cat_files):
    """【高速化】1枚の画像を生成する処理（マルチプロセス用ワーカー関数）"""
    # エラー回避のため、画像が読み込めるまでリトライ
    for _ in range(5):
        bg = cv2.imread(random.choice(bg_files))
        cat = cv2.imread(random.choice(cat_files), cv2.IMREAD_UNCHANGED)
        if bg is not None and cat is not None:
            break
    else:
        return False  # 読み込み失敗

    bg_h, bg_w = bg.shape[:2]

    # --- 1. サイズ変更 ---
    scale = random.uniform(0.5, 1.2)
    cat = cv2.resize(cat, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # --- 2. 反転と回転処理 ---
    if random.random() > 0.5:
        cat = cv2.flip(cat, 1)

    angle = random.uniform(-2.5, 2.5)
    cat = rotate_and_crop(cat, angle)

    cat_h, cat_w = cat.shape[:2]
    if cat_h >= bg_h or cat_w >= bg_w:
        scale_down = min((bg_h - 10) / cat_h, (bg_w - 10) / cat_w)
        cat = cv2.resize(
            cat, None, fx=scale_down, fy=scale_down, interpolation=cv2.INTER_AREA
        )
        cat_h, cat_w = cat.shape[:2]

    # --- 3. 街灯エフェクト ---
    if random.random() > 0.5:
        cat = apply_street_light_effect(cat)

    # --- 4. 出現位置 ---
    x_max = max(0, bg_w - cat_w)
    x = random.randint(0, x_max)

    y_min = bg_h // 2
    y_max = max(y_min, bg_h - cat_h)
    if y_min > y_max:
        y_min = y_max
    y = random.randint(y_min, y_max)

    # --- 5. アルファブレンド (【高速化】ブロードキャストによる行列計算) ---
    # [:, :, 3:]とすることで (H, W, 1) の形状を維持し、NumPyの高速な行列演算を可能にする
    alpha_cat = cat[:, :, 3:] / 255.0
    alpha_bg = 1.0 - alpha_cat

    # forループを使わずに一括でRGBを合成
    bg[y : y + cat_h, x : x + cat_w] = (
        alpha_cat * cat[:, :, :3] + alpha_bg * bg[y : y + cat_h, x : x + cat_w]
    ).astype(np.uint8)

    # --- 6. ノイズ適用 ---
    bg = add_iso_noise(bg, intensity_range=(0.02, 0.04))

    # --- 7. 保存 ---
    cx = (x + cat_w / 2) / bg_w
    cy = (y + cat_h / 2) / bg_h
    nw = cat_w / bg_w
    nh = cat_h / bg_h

    file_name = f"{idx:04d}"
    cv2.imwrite(os.path.join(SAVE_DIR, "images", f"{file_name}.jpg"), bg)

    with open(os.path.join(SAVE_DIR, "labels", f"{file_name}.txt"), "w") as f:
        f.write(f"{CLASS_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    return True


def augment_and_paste():
    bg_files = glob.glob(os.path.join(BG_DIR, "*.png")) + glob.glob(
        os.path.join(BG_DIR, "*.jpg")
    )
    cat_files = glob.glob(os.path.join(CAT_DIR, "*.png"))

    if not bg_files or not cat_files:
        print("エラー: 画像が見つかりません。パスを確認してください。")
        return

    print(f"{NUM_GENERATE}枚の画像データを生成中...")

    # ProcessPoolExecutorで並列処理を実行
    success_count = 0
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # idx(0〜999) をワーカー関数に渡して非同期実行
        futures = [
            executor.submit(generate_single_sample, i, bg_files, cat_files)
            for i in range(NUM_GENERATE)
        ]

        for future in concurrent.futures.as_completed(futures):
            if future.result():
                success_count += 1
                if success_count % 100 == 0:
                    print(f"  ... {success_count}/{NUM_GENERATE} 枚完了")

    print(f"完了: {SAVE_DIR} に保存されました。")


if __name__ == "__main__":
    augment_and_paste()
