import argparse
import importlib.machinery
import importlib.util
import json
import site
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

from inference import VALID_PRESET, apply_preprocess, detect_cat
from main import CLASS_ID, CONFIDENCE, MODEL_PATH

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def import_optuna_package():
    """ローカルの optuna.py と衝突しないように site-packages 側を優先して読み込む。"""
    search_paths = []
    try:
        search_paths.extend(site.getsitepackages())
    except Exception:
        pass

    try:
        user_site = site.getusersitepackages()
        if isinstance(user_site, str) and user_site:
            search_paths.append(user_site)
    except Exception:
        pass

    unique_paths = []
    seen = set()
    for path in search_paths:
        normalized = str(Path(path).resolve())
        if normalized not in seen:
            seen.add(normalized)
            unique_paths.append(path)

    spec = importlib.machinery.PathFinder.find_spec("optuna", unique_paths)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(
            "optuna パッケージが見つかりません。`uv add optuna` または `uv sync` を実行してください。"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules["optuna"] = module
    spec.loader.exec_module(module)
    return module


def load_images(directory: Path):
    images = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        images.append((path.name, frame))
    return images


def evaluate(model, with_cat_images, without_cat_images, params, imgsz):
    tp = 0
    fn = 0
    tn = 0
    fp = 0

    conf_threshold = params["confidence"]

    for _, frame in with_cat_images:
        target = apply_preprocess(frame, params)
        detected, _, _ = detect_cat(
            model,
            target,
            class_id=CLASS_ID,
            conf_threshold=conf_threshold,
            imgsz=imgsz,
        )
        if detected:
            tp += 1
        else:
            fn += 1

    for _, frame in without_cat_images:
        target = apply_preprocess(frame, params)
        detected, _, _ = detect_cat(
            model,
            target,
            class_id=CLASS_ID,
            conf_threshold=conf_threshold,
            imgsz=imgsz,
        )
        if detected:
            fp += 1
        else:
            tn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def build_trial_params(trial):
    return {
        "alpha": trial.suggest_float("alpha", 0.7, 1.4),
        "beta": trial.suggest_int("beta", -40, 40),
        "gamma": trial.suggest_float("gamma", 0.7, 1.6),
        "use_clahe": trial.suggest_categorical("use_clahe", [False, True]),
        "clahe_clip": trial.suggest_float("clahe_clip", 1.0, 4.0),
        "clahe_tile": trial.suggest_int("clahe_tile", 4, 12),
        "blur_ksize": trial.suggest_categorical("blur_ksize", [1, 3, 5]),
        "confidence": trial.suggest_float("confidence", 0.1, 0.6),
    }


def main():
    optuna = import_optuna_package()

    parser = argparse.ArgumentParser(
        description="Optunaで前処理パラメータを最適化して猫検出精度を上げる"
    )
    parser.add_argument("--trials", type=int, default=30, help="試行回数")
    parser.add_argument("--imgsz", type=int, default=640, help="推論時の入力サイズ")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("optuna_best_params.json"),
        help="最適化結果の保存先",
    )
    args = parser.parse_args()

    dataset_dir = Path("dataset")
    with_cat_dir = dataset_dir / "with_cat"
    without_cat_dir = dataset_dir / "without_cat"

    if not with_cat_dir.exists() or not without_cat_dir.exists():
        raise FileNotFoundError(
            f"dataset/with_cat または dataset/without_cat が見つかりません: {dataset_dir}"
        )

    with_cat_images = load_images(with_cat_dir)
    without_cat_images = load_images(without_cat_dir)

    if not with_cat_images or not without_cat_images:
        raise RuntimeError(
            "学習用画像が不足しています。with_cat と without_cat を確認してください。"
        )

    print("モデルをロードしています...")
    model = YOLO(MODEL_PATH, task="detect")

    base_params = VALID_PRESET.copy()
    base_params["confidence"] = CONFIDENCE
    base_metrics = evaluate(
        model=model,
        with_cat_images=with_cat_images,
        without_cat_images=without_cat_images,
        params=base_params,
        imgsz=args.imgsz,
    )
    print("\n[Baseline]")
    print(
        f"Accuracy={base_metrics['accuracy']:.2%}, Precision={base_metrics['precision']:.2%}, "
        f"Recall={base_metrics['recall']:.2%}, F1={base_metrics['f1']:.2%}"
    )

    def objective(trial):
        params = build_trial_params(trial)
        metrics = evaluate(
            model=model,
            with_cat_images=with_cat_images,
            without_cat_images=without_cat_images,
            params=params,
            imgsz=args.imgsz,
        )

        trial.set_user_attr("metrics", metrics)
        return metrics["f1"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)

    best_metrics = study.best_trial.user_attrs["metrics"]
    best_result = {
        "best_trial": study.best_trial.number,
        "best_value_f1": study.best_value,
        "best_params": study.best_params,
        "best_metrics": best_metrics,
        "baseline_metrics": base_metrics,
    }

    args.output.write_text(
        json.dumps(best_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n[Best Trial]")
    print(f"trial={study.best_trial.number}, F1={study.best_value:.2%}")
    print(
        f"Accuracy={best_metrics['accuracy']:.2%}, Precision={best_metrics['precision']:.2%}, "
        f"Recall={best_metrics['recall']:.2%}"
    )
    print(f"best_params={study.best_params}")
    print(f"結果を保存しました: {args.output}")


if __name__ == "__main__":
    main()
