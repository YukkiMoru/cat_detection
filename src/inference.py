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


def _resolve_model_path(path: Path) -> Path:
    """旧形式 `_size320_` のようなモデル名を、実在する `_size320x192_` 等へ寄せる。"""

    if path.exists():
        return path

    # 例: yolo26n_size320_mnn_fp16.mnn
    legacy = re.match(r"^(?P<prefix>.+?)_size(?P<w>\d+)_(?P<rest>.+)$", path.stem)
    if not legacy:
        return path

    prefix = legacy.group("prefix")
    w = legacy.group("w")
    rest = legacy.group("rest")
    ext = path.suffix

    candidates = sorted(path.parent.glob(f"{prefix}_size{w}x*_{rest}{ext}"))
    if candidates:
        logging.warning(
            f"指定モデルが見つからないため '{candidates[0].as_posix()}' を代わりに使用します。"
        )
        return candidates[0]

    return path


class CatDetector:
    def __init__(self, model_path: Path | None = None, params_path: Path | None = None):
        raw_model_path = (
            Path(model_path) if model_path is not None else Path(config.MODEL_PATH)
        )
        if not raw_model_path.is_absolute():
            raw_model_path = config.PROJECT_ROOT / raw_model_path

        self.model_path = _resolve_model_path(raw_model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"モデルが見つかりません: {self.model_path}")

        self._lut_cache_key: tuple[float, int, float] | None = None
        self._lut_cache_table: np.ndarray | None = None
        self._clahe_cache_key: tuple[float, int] | None = None
        self._clahe: Any | None = None

        # 例: yolo26n_size320x192_mnn_fp16.mnn -> 320x192 を抽出
        match = re.search(r"_size(\d+)x(\d+)_", self.model_path.name)
        if match:
            self.imgsz_w = int(match.group(1))
            self.imgsz_h = int(match.group(2))
            logging.debug(
                f"ファイル名から推論サイズ {self.imgsz_w}x{self.imgsz_h} を検出しました。"
            )
        else:
            # 例: *_size320_* のような中途半端な指定が来た場合は正方形として扱う
            match_legacy = re.search(r"_size(\d+)_", self.model_path.name)
            if match_legacy:
                self.imgsz_w = self.imgsz_h = int(match_legacy.group(1))
            else:
                self.imgsz_w = self.imgsz_h = config.IMGSZ
            logging.debug(
                f"サイズ指定が見つからないため、推論サイズ {self.imgsz_w}x{self.imgsz_h} を使用します。"
            )

        self.imgsz = f"{self.imgsz_w}x{self.imgsz_h}"

        self._configure_runtime_threads()

        try:
            # .pt, .onnx, .mnn いずれもこの1行でロード可能
            self.model = YOLO(str(self.model_path), task="detect")
        except Exception as e:
            logging.error(f"モデルのロードに失敗しました ({self.model_path.name}): {e}")
            raise

        # params_path が未指定なら、自分の名前のJSONを自動で探す
        if params_path is None:
            auto_params_path = (
                config.PROJECT_ROOT / "best_params" / f"{self.model_path.stem}.json"
            )
            if auto_params_path.exists():
                params_path = auto_params_path
                logging.debug(
                    f"専用パラメータ {auto_params_path.name} を自動検出しました。"
                )
            else:
                logging.debug(
                    "専用パラメータが見つからないためデフォルト設定を使用します。"
                )

        self.params = load_best_params(params_path)

    def _configure_runtime_threads(self) -> None:
        """Raspi4 などのCPU環境で推論が単一スレッドにならないよう明示する。"""

        num_threads = int(os.environ.get("CAT_DETECT_THREADS", os.cpu_count() or 4))
        num_threads = max(1, num_threads)

        try:
            cv2.setNumThreads(num_threads)
        except Exception:
            pass

        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            os.environ.setdefault(key, str(num_threads))

        if self.model_path.suffix.lower() == ".mnn":
            os.environ.setdefault("MNN_NUM_THREADS", str(num_threads))

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

        # 重要: 高解像度のまま前処理をせず、先に推論サイズへ縮小してから実施する
        processed = cv2.resize(
            frame,
            (self.imgsz_w, self.imgsz_h),
            interpolation=cv2.INTER_AREA,
        )

        # convertScaleAbs(明るさ/コントラスト) + gamma を 1回の LUT に統合
        lut_key = (alpha, beta, gamma)
        if self._lut_cache_key != lut_key or self._lut_cache_table is None:
            x = np.arange(256, dtype=np.float32)
            y = np.abs(x * alpha + float(beta))
            y = np.clip(y, 0.0, 255.0)
            if abs(gamma - 1.0) > 1e-6:
                inv_gamma = 1.0 / float(gamma)
                y = ((y / 255.0) ** inv_gamma) * 255.0
            self._lut_cache_table = y.astype(np.uint8)
            self._lut_cache_key = lut_key

        if not (abs(alpha - 1.0) < 1e-6 and beta == 0 and abs(gamma - 1.0) < 1e-6):
            processed = cv2.LUT(processed, self._lut_cache_table)

        if use_clahe:
            clahe_tile = max(1, min(clahe_tile, 4))
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)

            clahe_key = (clahe_clip, clahe_tile)
            if self._clahe_cache_key != clahe_key or self._clahe is None:
                self._clahe = cv2.createCLAHE(
                    clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile)
                )
                self._clahe_cache_key = clahe_key

            l_channel = self._clahe.apply(l_channel)
            processed = cv2.cvtColor(
                cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR
            )

        if blur_ksize > 1:
            processed = cv2.GaussianBlur(processed, (blur_ksize, blur_ksize), 0)

        return processed

    def detect_cat(self, frame) -> Tuple[bool, float, List[Any]]:
        """前処理を適用したうえでYOLO推論を行い、猫クラスの検出結果を返す。"""

        processed = self.apply_preprocess(frame)
        conf_threshold = self.params.get("confidence", config.CONFIDENCE)

        detected = False
        max_conf = 0.0
        results: List[Any] = []

        try:
            results = self.model(
                processed,
                classes=[config.CLASS_ID],
                conf=conf_threshold,
                verbose=False,
                imgsz=[self.imgsz_h, self.imgsz_w],
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
    test_model_path = None  # None の場合は config.MODEL_PATH が使われます

    if not img_path.exists():
        print(f"画像が見つかりません: {img_path}")
        raise SystemExit(1)

    print("モデルをロードしています...")
    detector = CatDetector(model_path=test_model_path)

    frame = cv2.imread(str(img_path))
    if frame is None:
        print("画像の読み込みに失敗しました。")
        raise SystemExit(1)

    print(f"ロードされたモデル: {detector.model_path.name}")
    print(f"推論サイズ: {detector.imgsz_w}x{detector.imgsz_h}")
    print(f"使用パラメータ: {detector.params}")

    processed = detector.apply_preprocess(frame)
    detected, max_conf, results = detector.detect_cat(frame)

    print(f"検出結果: {detected}, 最大信頼度: {max_conf:.2f}")

    if results and detected:
        processed_with_boxes = results[0].plot()
    else:
        processed_with_boxes = processed.copy()

    orig_resized = cv2.resize(
        frame,
        (processed.shape[1], processed.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    concat_img = cv2.hconcat([orig_resized, processed_with_boxes])

    cv2.imshow("Original | Processed+Detection (Press Any Key to Close)", concat_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
