import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

import config


def load_best_params(path: Path | None = None) -> Dict:
    target_path = config.BEST_PARAMS_PATH if path is None else Path(path)
    if not target_path.is_absolute():
        target_path = config.PROJECT_ROOT / target_path

    if not target_path.exists():
        return config.VALID_PRESET.copy()

    try:
        data = json.loads(target_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return config.VALID_PRESET.copy()

        best_params = data.get("best_params", data)
        if not isinstance(best_params, dict):
            return config.VALID_PRESET.copy()

        merged_params = config.VALID_PRESET.copy()
        merged_params.update(best_params)
        return merged_params
    except Exception:
        return config.VALID_PRESET.copy()


class CatDetector:
    def __init__(self, model_path: Path | None = None, params_path: Path | None = None):
        self.model_path = (
            Path(model_path) if model_path is not None else Path(config.MODEL_PATH)
        )

        # 例: yolo26n_size320_mnn_fp16.mnn -> 320 を抽出
        match = re.search(r"_size(\d+)_", self.model_path.name)
        if match:
            self.imgsz = int(match.group(1))
            logging.debug(f"ファイル名から推論サイズ {self.imgsz} を検出しました。")
        else:
            self.imgsz = config.IMGSZ
            logging.debug(
                f"サイズ指定が見つからないため、config.IMGSZ ({self.imgsz}) を使用します。"
            )

        try:
            # .pt, .onnx, .mnn いずれもこの1行でロード可能
            self.model = YOLO(str(self.model_path), task="detect")
        except Exception as e:
            logging.error(f"モデルのロードに失敗しました ({self.model_path.name}): {e}")
            raise

        # --- 改良ポイント: params_path が未指定なら、自分の名前のJSONを自動で探す ---
        if params_path is None:
            auto_params_path = Path("best_params") / f"{self.model_path.stem}.json"
            if auto_params_path.exists():
                params_path = auto_params_path
                logging.debug(f"専用パラメータ {params_path.name} を自動検出しました。")
            else:
                logging.debug(
                    "専用パラメータが見つからないためデフォルト設定を使用します。"
                )

        self.params = load_best_params(params_path)

    def set_params(self, params: Dict | None = None) -> None:
        self.params = config.VALID_PRESET.copy()
        if params:
            self.params.update(params)

    def apply_preprocess(self, frame):
        params = self.params
        alpha = float(params.get("alpha", config.VALID_PRESET["alpha"]))
        beta = int(params.get("beta", config.VALID_PRESET["beta"]))
        gamma = float(params.get("gamma", config.VALID_PRESET["gamma"]))
        use_clahe = bool(params.get("use_clahe", config.VALID_PRESET["use_clahe"]))
        clahe_clip = float(params.get("clahe_clip", config.VALID_PRESET["clahe_clip"]))
        clahe_tile = int(params.get("clahe_tile", config.VALID_PRESET["clahe_tile"]))
        blur_ksize = int(params.get("blur_ksize", config.VALID_PRESET["blur_ksize"]))

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

    def detect_cat(self, frame) -> Tuple[bool, float, List[Any]]:
        """
        前処理を適用したうえでYOLO推論を行い、猫クラスの検出結果を返す。
        戻り値: (detected(bool), max_conf(float), results(list))
        """
        processed = self.apply_preprocess(frame)
        conf_threshold = self.params.get("confidence", config.CONFIDENCE)

        detected = False
        max_conf = 0.0
        results = []

        try:
            results = self.model(
                processed,
                classes=[config.CLASS_ID],
                conf=conf_threshold,
                verbose=False,
                imgsz=self.imgsz,
            )
            if (
                results
                and getattr(results[0], "boxes", None)
                and len(results[0].boxes) > 0
            ):
                detected = True
                conf_val = results[0].boxes.conf.max()
                max_conf = float(
                    conf_val.item() if hasattr(conf_val, "item") else conf_val
                )
        except Exception as e:
            logging.error(f"モデル推論エラー: {e}")

        return detected, max_conf, results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    img_path = Path("dataset/with_cat/P1.jpg")

    # テストとして、エクスポート済みのモデルなどを直接指定して動作確認できます
    test_model_path = None  # None の場合は config.MODEL_PATH が使われます

    if not img_path.exists():
        print(f"画像が見つかりません: {img_path}")
    else:
        print("モデルをロードしています...")
        detector = CatDetector(model_path=test_model_path)

        frame = cv2.imread(str(img_path))
        if frame is None:
            print("画像の読み込みに失敗しました。")
        else:
            print(f"ロードされたモデル: {detector.model_path.name}")
            print(f"推論サイズ: {detector.imgsz}")
            print(f"使用パラメータ: {detector.params}")

            processed = detector.apply_preprocess(frame)
            detected, max_conf, results = detector.detect_cat(frame)

            print(f"検出結果: {detected}, 最大信頼度: {max_conf:.2f}")

            if results and detected:
                img_to_show = results[0].plot()
            else:
                img_to_show = processed

            cv2.imshow("Processed & Detected (Press Any Key to Close)", img_to_show)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
