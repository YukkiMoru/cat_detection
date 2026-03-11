from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BEST_PARAMS_PATH = PROJECT_ROOT / "optuna_best_params.json"

HEADLESS = False
CAMERA_ID = 0
FPS = 5
MODEL_PATH = PROJECT_ROOT / "models" / "yolo26n.onnx"
CONFIDENCE = 0.2
CLASS_ID = 15
IMGSZ = 640
DURATION_THRESH = 1.0
RESET_THRESH = 5.0

VALID_PRESET = {
    "alpha": 1.5,        # コントラスト：1.0〜2.0が一般的
    "beta": 10,          # 明るさ補正：整数で扱いやすく
    "gamma": 1.0,        # ガンマ：1.0（等倍）を基準に調整
    "use_clahe": True,
    "clahe_clip": 2.0,   # CLAHEの標準：2.0〜4.0が一般的（1.2はかなり弱め）
    "clahe_tile": 8,     # 8x8分割がOpenCV等のデフォルトで最も一般的
    "blur_ksize": 5,     # そのまま（ノイズ除去に適切なサイズ）
    "confidence": 0.5,   # 信頼度：0.5（50%）を基準にするのが定石
}
