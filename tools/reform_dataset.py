#!/usr/bin/env python
# uv run view_dataset.py

import shutil
from pathlib import Path

import cv2

# ==========================================
# ⚙️ 設定（検証用データ構築用）
# ==========================================
DATASET_DIRS = [Path("dataset_boxed/original")]  # 元データ
EXPORT_DIR = Path("dataset_boxed2/original")  # 選別後の保存先
CLASS_ID = 15  # 対象のクラスID
# ==========================================

# --- グローバル変数（マウス操作用） ---
drawing = False
ix, iy = -1, -1
temp_bbox = None  # [x1, y1, x2, y2]


def mouse_callback(event, x, y, flags, param):
    """マウス操作によるボックス描画の制御"""
    global ix, iy, drawing, temp_bbox
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing, ix, iy = True, x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp_bbox = [ix, iy, x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        temp_bbox = [ix, iy, x, y]


# --- ヘルパー関数 ---
def yolo_to_pixel(parts, w, h):
    """YOLO形式をピクセル座標[class_id, x1, y1, x2, y2]に変換"""
    cx, cy, bw, bh = map(float, parts[1:5])
    x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
    x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
    return [int(parts[0]), x1, y1, x2, y2]


def pixel_to_yolo(box, w, h, class_id):
    """ピクセル座標をYOLO形式の文字列に変換"""
    x1, y1, x2, y2 = box
    nx1, nx2 = sorted([x1, x2])  # 逆から描画した場合の対策
    ny1, ny2 = sorted([y1, y2])
    cx, cy = (nx1 + nx2) / 2 / w, (ny1 + ny2) / 2 / h
    bw, bh = (nx2 - nx1) / w, (ny2 - ny1) / h
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def save_current_data(img_path, active_box, w, h, export_label):
    """画像とラベルを保存する共通処理"""
    save_img = EXPORT_DIR / "images" / img_path.name
    # 画像コピー：未保存の場合のみ実行
    if not save_img.exists():
        shutil.copy(img_path, save_img)
    # ラベル書き込み
    with open(export_label, "w") as f:
        if active_box:
            f.write(pixel_to_yolo(active_box, w, h, CLASS_ID))


def draw_ui_panel(img, current, total, filename, has_box, is_edited, saved_msg=""):
    """上部の操作ガイドUIを描画"""
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    font, LT = cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA

    cv2.putText(
        img, "Auto-Save on [R] [T] [S]", (30, 40), font, 1.0, (0, 255, 0), 3, LT
    )
    cv2.putText(
        img,
        "[D] Delete Box  [T] Next  [R] Prev  [Q] Exit",
        (30, 75),
        font,
        0.8,
        (255, 255, 255),
        2,
        LT,
    )

    box_status = "BOX: OK" if has_box else "BOX: EMPTY"
    edit_status = "(COMPLETED)" if is_edited else "(ORIGINAL)"
    color = (0, 255, 255) if is_edited else (200, 200, 200)
    cv2.putText(
        img,
        f"{current}/{total} | {box_status} {edit_status} | {filename}",
        (30, 110),
        font,
        0.6,
        color,
        2,
        LT,
    )

    if saved_msg:
        cv2.putText(
            img, saved_msg, (w // 2 - 150, h // 2 + 50), font, 3.0, (0, 255, 0), 6, LT
        )


def main():
    global temp_bbox
    image_paths = sorted(
        [
            p
            for d_dir in DATASET_DIRS
            for ext in ("*.jpg", "*.jpeg", "*.png")
            for p in (d_dir / "images").glob(ext)
            if (d_dir / "images").exists()
        ]
    )

    if not image_paths:
        print(f"❌ 画像が見つかりません: {DATASET_DIRS}")
        return

    (EXPORT_DIR / "images").mkdir(parents=True, exist_ok=True)
    (EXPORT_DIR / "labels").mkdir(parents=True, exist_ok=True)

    current_idx, msg_timer, saved_msg = 0, 0, ""
    win_name = "Dataset Editor"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, mouse_callback)

    while True:
        img_path = image_paths[current_idx]
        orig_label = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
        export_label = EXPORT_DIR / "labels" / f"{img_path.stem}.txt"

        is_edited = export_label.exists()
        target_label = export_label if is_edited else orig_label
        img_orig = cv2.imread(str(img_path))
        if img_orig is None:
            current_idx = (current_idx + 1) % len(image_paths)
            continue

        h, w = img_orig.shape[:2]
        current_box = None
        if target_label.exists():
            with open(target_label, "r") as f:
                parts = f.readline().strip().split()
                if len(parts) >= 5:
                    current_box = yolo_to_pixel(parts, w, h)

        while True:
            display_img = img_orig.copy()
            active_box = temp_bbox or (current_box[1:] if current_box else None)

            if temp_bbox:
                cv2.rectangle(
                    display_img,
                    (temp_bbox[0], temp_bbox[1]),
                    (temp_bbox[2], temp_bbox[3]),
                    (0, 0, 255),
                    3,
                )
            elif current_box:
                cv2.rectangle(
                    display_img,
                    (current_box[1], current_box[2]),
                    (current_box[3], current_box[4]),
                    (0, 255, 0),
                    2,
                )

            draw_ui_panel(
                display_img,
                current_idx + 1,
                len(image_paths),
                img_path.name,
                active_box is not None,
                is_edited,
                saved_msg,
            )
            cv2.imshow(win_name, display_img)

            if msg_timer > 0:
                msg_timer -= 1
            else:
                saved_msg = ""

            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                return

            # --- 保存して移動する処理 (S, T, R キー) ---
            if key in (ord("s"), ord("t"), ord("r")):
                # 現在の状態を保存
                save_current_data(img_path, active_box, w, h, export_label)

                if key == ord("s"):
                    saved_msg, msg_timer = "SAVED!", 15
                    # Sキーの場合は次の画像へ進む
                    current_idx = (current_idx + 1) % len(image_paths)
                elif key == ord("t"):
                    current_idx = (current_idx + 1) % len(image_paths)
                elif key == ord("r"):
                    current_idx = (current_idx - 1) % len(image_paths)

                temp_bbox = None
                break

            if key == ord("d"):
                current_box, temp_bbox = None, None
                print("🗑️ ボックスを削除しました（保存時に反映されます）")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
