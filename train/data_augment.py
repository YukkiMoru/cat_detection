#!/usr/bin/env python
import glob
import os
import random
import shutil

import cv2
import numpy as np
from tqdm import tqdm

# --- 設定 ---
# 1. 元の画像（素材）が入っているフォルダ
BASE_DIR = "train/dataset_boxed"

# 2. 生成したデータセットを書き出すフォルダ
SAVE_BASE = "train/dataset_generated"

# 素材のパス
POS_IMG_DIR = os.path.join(BASE_DIR, "original/cat/images")
POS_LBL_DIR = os.path.join(BASE_DIR, "original/cat/labels")
NEG_IMG_DIR = os.path.join(BASE_DIR, "original/not/images")
NEG_LBL_DIR = os.path.join(BASE_DIR, "original/not/labels")
COLLAGE_DIR = os.path.join(BASE_DIR, "collage")

NUM_SYNTHETIC = 3000  # 合成画像の割合を高めるため、合成枚数を大幅に増加
REAL_OVERSAMPLE = 20  # 実画像が合成画像に負けないように増やす
VAL_RATIO = 0.2
TEST_RATIO = 0.1
CLASS_ID = 15
RANDOM_SEED = 42


def setup_dirs():
    # SAVE_BASE（生成先）だけを削除・再作成するので、BASE_DIR（素材）は無事です
    if os.path.exists(SAVE_BASE):
        shutil.rmtree(SAVE_BASE)
    for split in ["train", "val", "test"]:
        for d in ["images", "labels"]:
            os.makedirs(os.path.join(SAVE_BASE, split, d), exist_ok=True)


def rotate_img(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101
    )


def get_images(target_dir):
    # Windows/Linux両方の拡張子に対応
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(target_dir, e)))
    return files


def save_sample(img, label_lines, split, name):
    cv2.imwrite(os.path.join(SAVE_BASE, split, "images", f"{name}.jpg"), img)
    with open(os.path.join(SAVE_BASE, split, "labels", f"{name}.txt"), "w") as f:
        f.writelines(label_lines)


def split_cat_and_alpha(cat_img):
    """Return BGR image and alpha mask in [0, 1] for both BGRA/BGR inputs."""
    if cat_img is None:
        return None, None

    if len(cat_img.shape) == 2:
        cat_bgr = cv2.cvtColor(cat_img, cv2.COLOR_GRAY2BGR)
        alpha = np.ones(cat_img.shape, dtype=np.float32)
        return cat_bgr, alpha

    if cat_img.shape[2] == 4:
        cat_bgr = cat_img[:, :, :3]
        alpha = cat_img[:, :, 3].astype(np.float32) / 255.0
        return cat_bgr, alpha

    if cat_img.shape[2] == 3:
        alpha = np.ones(cat_img.shape[:2], dtype=np.float32)
        return cat_img, alpha

    return None, None


def main():
    # 全体用の初期シード設定
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    setup_dirs()

    # 1. 素材の読み込み
    pos_files = get_images(POS_IMG_DIR)
    neg_files = get_images(NEG_IMG_DIR)

    print(f"📊 素材集計: 猫あり={len(pos_files)}枚 / 猫なし={len(neg_files)}枚")

    if len(pos_files) == 0 or len(neg_files) == 0:
        print("❌ エラー: 画像が見つかりません。")
        print(f"確認してください -> {POS_IMG_DIR}")
        return

    # 分割ロジック
    def get_split_lists(lst):
        # シード固定されているため、毎回同じ分配になる
        random.shuffle(lst)
        n = len(lst)
        test_n = int(n * TEST_RATIO)
        val_n = int(n * VAL_RATIO)
        return lst[test_n + val_n :], lst[test_n : test_n + val_n], lst[:test_n]

    tr_p, va_p, te_p = get_split_lists(pos_files)
    tr_n, va_n, te_n = get_split_lists(neg_files)

    # 2. Train / Val / Test の書き出し
    for split_name, p_list, n_list in [
        ("train", tr_p, tr_n),
        ("val", va_p, va_n),
        ("test", te_p, te_n),
    ]:
        print(f"🚚 {split_name}セットを作成中...")
        # 猫あり
        for f in tqdm(p_list, desc="Pos"):
            img = cv2.imread(f)
            name = os.path.splitext(os.path.basename(f))[0]
            lbl_path = os.path.join(POS_LBL_DIR, f"{name}.txt")
            lines = []
            if os.path.exists(lbl_path):
                with open(lbl_path, "r") as fl:
                    for l in fl:
                        p = l.split()
                        if len(p) >= 5:
                            lines.append(f"{CLASS_ID} {p[1]} {p[2]} {p[3]} {p[4]}\n")

            count = REAL_OVERSAMPLE if split_name == "train" else 1
            for c in range(count):
                aug = (
                    rotate_img(img, random.uniform(-5, 5))
                    if (c > 0 and split_name == "train")
                    else img
                )
                save_sample(aug, lines, split_name, f"real_p_{name}_{c:02d}")
        # 猫なし
        for f in tqdm(n_list, desc="Neg"):
            save_sample(
                cv2.imread(f),
                [],
                split_name,
                f"real_n_{os.path.splitext(os.path.basename(f))[0]}",
            )

    # 3. 合成画像の追加（背景として tr_n を使用）
    print(f"🎨 合成画像を {NUM_SYNTHETIC} 枚生成中...")
    cats = get_images(COLLAGE_DIR)
    if len(cats) == 0:
        print(f"⚠️ 警告: コラージュ素材が見つかりません -> {COLLAGE_DIR}")
        return

    for i in tqdm(range(NUM_SYNTHETIC)):
        # 実写枚数の増減等で乱数消費回数が変わっても、i番目の合成処理は必ず同じになる
        loop_seed = RANDOM_SEED + i + 10000
        random.seed(loop_seed)
        np.random.seed(loop_seed)

        bg_path = random.choice(tr_n)
        bg_raw = cv2.imread(bg_path)
        if bg_raw is None:
            continue

        # 背景を ±5度 ランダム回転
        bg = rotate_img(bg_raw, random.uniform(-5, 5))

        cat = cv2.imread(random.choice(cats), cv2.IMREAD_UNCHANGED)
        cat_bgr, alpha = split_cat_and_alpha(cat)
        if cat_bgr is None or alpha is None:
            continue

        s = random.uniform(0.8, 1.4)  # 合成画像の猫の大きさ
        cat_bgr = cv2.resize(cat_bgr, None, fx=s, fy=s)
        alpha = cv2.resize(alpha, None, fx=s, fy=s)

        # 猫を ±3度 ランダム回転
        cat_angle = random.uniform(-3, 3)
        cat_bgr = rotate_img(cat_bgr, cat_angle)
        alpha = rotate_img(alpha, cat_angle)

        h_b, w_b = bg.shape[:2]
        h_c, w_c = cat_bgr.shape[:2]
        if h_b <= h_c or w_b <= w_c:
            continue

        x, y = random.randint(0, w_b - w_c), random.randint(h_b // 4, h_b - h_c)
        alpha = cv2.GaussianBlur(alpha.astype(np.float32), (3, 3), 0)
        alpha = np.clip(alpha, 0.0, 1.0)

        for c in range(3):
            bg[y : y + h_c, x : x + w_c, c] = (
                alpha * cat_bgr[:, :, c] + (1 - alpha) * bg[y : y + h_c, x : x + w_c, c]
            )

        line = [
            f"{CLASS_ID} {(x + w_c / 2) / w_b:.6f} {(y + h_c / 2) / h_b:.6f} {w_c / w_b:.6f} {h_c / h_b:.6f}\n"
        ]
        save_sample(bg, line, "train", f"synth_{i:05d}")

    # 合成ループが終了したらシードを元に戻す（後続の処理に影響を与えないため）
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)


if __name__ == "__main__":
    main()
