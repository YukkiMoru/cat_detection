import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
from ultralytics import YOLO

import config


def load_best_params(path: Path | None = None) -> Dict:
    target_path = config.BEST_PARAMS_PATH if path is None else Path(path)
    if not target_path.suffix == ".json":
        # パスがディレクトリを指している場合はデフォルトを返す
        return config.VALID_PRESET.copy()

    if not target_path.exists():
        return config.VALID_PRESET.copy()

    try:
        data = json.loads(target_path.read_text(encoding="utf-8"))
        best_params = data.get("best_params", data)
        merged_params = config.VALID_PRESET.copy()
        if isinstance(best_params, dict):
            merged_params.update(best_params)
        return merged_params
    except Exception:
        return config.VALID_PRESET.copy()


def _resolve_model_path(path: Path) -> Path:
    if path.exists():
        return path
    legacy = re.match(r"^(?P<prefix>.+?)_size(?P<w>\d+)_(?P<rest>.+)$", path.stem)
    if not legacy:
        return path

    candidates = sorted(
        path.parent.glob(
            f"{legacy.group('prefix')}_size{legacy.group('w')}x*_{legacy.group('rest')}{path.suffix}"
        )
    )
    return candidates[0] if candidates else path


class CatDetector:
    def __init__(self, model_path: Path | None = None, params_path: Path | None = None):
        raw_path = Path(model_path or config.MODEL_PATH)
        if not raw_path.is_absolute():
            raw_path = config.PROJECT_ROOT / raw_path

        self.model_path = _resolve_model_path(raw_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # サイズ抽出 (filename_size640x384_... 形式)
        match = re.search(r"_size(\d+)x(\d+)_", self.model_path.name)
        if match:
            self.imgsz_w, self.imgsz_h = int(match.group(1)), int(match.group(2))
        else:
            # 見つからない場合はconfigの値を採用
            self.imgsz_w = getattr(config, "IMGSZ_W", 640)
            self.imgsz_h = getattr(config, "IMGSZ_H", 640)

        self.imgsz = (self.imgsz_h, self.imgsz_w)  # YOLO用 (h, w)
        self._configure_runtime_threads()

        try:
            self.model = YOLO(str(self.model_path), task="detect")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

        # パラメータの自動解決
        if params_path is None:
            params_path = (
                config.PROJECT_ROOT / "best_params" / f"{self.model_path.stem}.json"
            )

        self.params = load_best_params(params_path)

    def _configure_runtime_threads(self):
        num_threads = int(os.environ.get("CAT_DETECT_THREADS", os.cpu_count() or 4))
        cv2.setNumThreads(num_threads)
        for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
            os.environ.setdefault(key, str(num_threads))

    def set_params(self, params: Dict | None = None):
        self.params = config.VALID_PRESET.copy()
        if params:
            self.params.update(params)

    def apply_preprocess(self, frame):
        """推論前処理は最小構成（リサイズのみ）にする。"""
        return cv2.resize(
            frame, (self.imgsz_w, self.imgsz_h), interpolation=cv2.INTER_AREA
        )

    def detect_cat(self, frame) -> Tuple[bool, float, List[Any]]:
        processed = self.apply_preprocess(frame)
        conf_threshold = self.params.get("confidence", 0.25)

        try:
            # imgsz に (h, w) を渡すことでモデル内の自動リサイズとアスペクト比を制御
            results = self.model(
                processed,
                classes=[config.CLASS_ID],
                conf=conf_threshold,
                verbose=False,
                imgsz=self.imgsz,
            )

            if results and len(results[0].boxes) > 0:
                max_conf = float(results[0].boxes.conf.max().item())
                return True, max_conf, results
        except Exception as e:
            logging.error(f"Inference Error: {e}")

        return False, 0.0, []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    detector = CatDetector()

    # テスト画像パス
    img_path = Path("dataset_boxed/val/images").glob("*.jpg")
    try:
        target = next(img_path)
        frame = cv2.imread(str(target))
        detected, conf, results = detector.detect_cat(frame)

        print(
            f"Model: {detector.model_path.name} | Size: {detector.imgsz_w}x{detector.imgsz_h}"
        )
        print(f"Detected: {detected} (Conf: {conf:.2f})")

        res_img = results[0].plot() if detected else detector.apply_preprocess(frame)
        cv2.imshow("Inference Test", res_img)
        cv2.waitKey(0)
    except StopIteration:
        print("No images found in dataset_boxed/val/images")
