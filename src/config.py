from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BEST_PARAMS_PATH = PROJECT_ROOT / "optuna_best_params.json"

HEADLESS = False
CAMERA_ID = 0
FPS = 5
MODEL_PATH = "models/best.pt"
CONFIDENCE = 0.2
CLASS_ID = 15
IMGSZ_W = 640
IMGSZ_H = 384
DURATION_THRESH = 1.0
RESET_THRESH = 5.0

VALID_PRESET = {
    "confidence": CONFIDENCE,
}
