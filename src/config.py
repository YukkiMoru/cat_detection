from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BEST_PARAMS_PATH = PROJECT_ROOT / "optuna_best_params.json"

HEADLESS = False
CAMERA_ID = 0
FPS = 5
MODEL_PATH = "models/yolo26n/yolo26n_size640_mnn_fp32.mnn"
CONFIDENCE = 0.2
CLASS_ID = 15
IMGSZ = 640
DURATION_THRESH = 1.0
RESET_THRESH = 5.0

VALID_PRESET = {
    "alpha": 0.9723211351452129,
    "beta": 3,
    "gamma": 1.5293564399958632,
    "use_clahe": True,
    "clahe_clip": 2.5514849604830268,
    "clahe_tile": 12,
    "blur_ksize": 5,
    "confidence": 0.10770572241125684,
}
