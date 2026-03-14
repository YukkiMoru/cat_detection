from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BEST_PARAMS_PATH = PROJECT_ROOT / "optuna_best_params.json"

HEADLESS = False
CAMERA_ID = 0
FPS = 5
MODEL_PATH = "models/yolov5su/yolov5su_size640x384_openvino_fp32_openvino_model"
CONFIDENCE = 0.2
CLASS_ID = 15
IMGSZ = 640
DURATION_THRESH = 1.0
RESET_THRESH = 5.0

VALID_PRESET = {
    "alpha": 1.3526395510689064,
    "beta": -3,
    "gamma": 0.90653121009541,
    "use_clahe": False,
    "clahe_clip": 2.7435010513797966,
    "clahe_tile": 4,
    "blur_ksize": 3,
    "confidence": 0.25984842563217536,
}
