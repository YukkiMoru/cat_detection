import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
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

        # Cache用
        self._lut_cache_key = None
        self._lut_cache_table = None
        self._clahe = None
        self._clahe_cache_key = None

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
        """歪みを抑えてリサイズし、前処理を適用"""
        # アスペクト比を維持してリサイズ (1920x1080 -> 640x360 -> パディングして 640x384 等)
        # 今回は簡易的に 16:9 -> 640x384 への直接リサイズ（歪みは最小限）
        processed = cv2.resize(
            frame, (self.imgsz_w, self.imgsz_h), interpolation=cv2.INTER_AREA
        )

        alpha = float(self.params.get("alpha", 1.0))
        beta = int(self.params.get("beta", 0))
        gamma = float(self.params.get("gamma", 1.0))

        # LUTによる高速なコントラスト/ガンマ補正
        lut_key = (alpha, beta, gamma)
        if self._lut_cache_key != lut_key:
            x = np.arange(256, dtype=np.float32)
            y = np.clip(x * alpha + beta, 0, 255)
            if abs(gamma - 1.0) > 1e-6:
                y = ((y / 255.0) ** (1.0 / gamma)) * 255.0
            self._lut_cache_table = y.astype(np.uint8)
            self._lut_cache_key = lut_key

        processed = cv2.LUT(processed, self._lut_cache_table)

        if self.params.get("use_clahe", False):
            clip = self.params.get("clahe_clip", 2.0)
            tile = self.params.get("clahe_tile", 4)
            if self._clahe_cache_key != (clip, tile):
                self._clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
                self._clahe_cache_key = (clip, tile)

            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self._clahe.apply(l)
            processed = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        ksize = int(self.params.get("blur_ksize", 1))
        if ksize > 1:
            processed = cv2.GaussianBlur(processed, (ksize, ksize), 0)

        return processed

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
