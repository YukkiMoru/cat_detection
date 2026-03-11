import argparse
import json
from pathlib import Path

import optuna
from ultralytics import YOLO

from inference import BEST_PARAMS_PATH, VALID_PRESET, load_best_params
from main import MODEL_PATH
from precision import evaluate, load_images


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
    parser = argparse.ArgumentParser(
        description="Optunaで前処理パラメータを最適化して猫検出精度を上げる"
    )
    parser.add_argument("--trials", type=int, default=30, help="試行回数")
    parser.add_argument("--imgsz", type=int, default=640, help="推論時の入力サイズ")
    parser.add_argument(
        "--output",
        type=Path,
        default=BEST_PARAMS_PATH,
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

    base_params = load_best_params()
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

    def stop_when_one(study, trial):
        if study.best_value is not None and study.best_value >= 0.9999:
            print("\nF1=1.0に到達したため最適化を停止します。")
            study.stop()

    study = optuna.create_study(direction="maximize")
    study.enqueue_trial(base_params)
    study.optimize(objective, n_trials=args.trials, callbacks=[stop_when_one])

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
