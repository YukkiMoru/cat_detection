import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

VALID_PRESET = {
    "alpha": 1.3752021028813055,
    "beta": 40,
    "gamma": 0.7077427684305686,
    "use_clahe": True,
    "clahe_clip": 2.1819227142399846,
    "clahe_tile": 8,
    "blur_ksize": 5,
    "confidence": 0.4264685958606309,
}


def load_best_params(path: Path = Path("optuna_best_params.json")) -> Dict:
    if not path.exists():
        return VALID_PRESET.copy()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # best_params may be nested under best_params
        return data.get("best_params", {}) if isinstance(data, dict) else {}
    except Exception:
        return VALID_PRESET.copy()


def apply_preprocess(frame, params: Dict = None):
    if params is None:
        params = VALID_PRESET

    alpha = float(params.get("alpha", VALID_PRESET["alpha"]))
    beta = int(params.get("beta", VALID_PRESET["beta"]))
    gamma = float(params.get("gamma", VALID_PRESET["gamma"]))
    use_clahe = bool(params.get("use_clahe", VALID_PRESET["use_clahe"]))
    clahe_clip = float(params.get("clahe_clip", VALID_PRESET["clahe_clip"]))
    clahe_tile = int(params.get("clahe_tile", VALID_PRESET["clahe_tile"]))
    blur_ksize = int(params.get("blur_ksize", VALID_PRESET["blur_ksize"]))

    processed = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    if abs(gamma - 1.0) > 1e-6:
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)],
            dtype=np.uint8,
        )
        processed = cv2.LUT(processed, table)

    if use_clahe:
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile)
        )
        l_channel = clahe.apply(l_channel)
        processed = cv2.cvtColor(
            cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR
        )

    if blur_ksize > 1:
        processed = cv2.GaussianBlur(processed, (blur_ksize, blur_ksize), 0)

    return processed


def detect_cat(
    model, frame, class_id=15, conf_threshold=None, imgsz=640
) -> Tuple[bool, float, List[Any]]:
    """
    YOLO推論を行い、指定クラスの検出結果を返す関数
    戻り値: (detected(bool), max_conf(float), results(list))
    """
    if conf_threshold is None:
        conf_threshold = VALID_PRESET["confidence"]

    detected = False
    max_conf = 0.0
    results = []

    try:
        results = model(
            frame, classes=[class_id], conf=conf_threshold, verbose=False, imgsz=imgsz
        )
        if results and getattr(results[0], "boxes", None) and len(results[0].boxes) > 0:
            detected = True
            conf_val = results[0].boxes.conf.max()
            max_conf = float(conf_val.item() if hasattr(conf_val, "item") else conf_val)
    except Exception as e:
        logging.error(f"モデル推論エラー: {e}")

    return detected, max_conf, results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    img_path = Path("dataset/with_cat/P1.jpg")

    if not img_path.exists():
        print(f"画像が見つかりません: {img_path}")
    else:
        model_path = "yolo26n.onnx"
        print(f"モデル {model_path} をロードしています...")
        model = YOLO(model_path, task="detect")

        frame = cv2.imread(str(img_path))
        if frame is None:
            print("画像の読み込みに失敗しました。")
        else:
            params = load_best_params()
            print(f"使用パラメータ: {params}")

            proc_frame = apply_preprocess(frame, params)

            conf = params.get("confidence", VALID_PRESET["confidence"])
            detected, max_conf, results = detect_cat(
                model, proc_frame, class_id=15, conf_threshold=conf, imgsz=640
            )

            print(f"検出結果: {detected}, 最大信頼度: {max_conf:.2f}")

            if results and detected:
                img_to_show = results[0].plot()
            else:
                img_to_show = proc_frame

            cv2.imshow("Processed & Detected (Press Any Key to Close)", img_to_show)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
