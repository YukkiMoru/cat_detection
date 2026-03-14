import argparse
import json
from pathlib import Path

import optuna

from inference import CatDetector, load_best_params
from precision import evaluate, load_image_paths


def build_trial_params(trial):
    return {
        "alpha": trial.suggest_float("alpha", 0.7, 1.4),
        "beta": trial.suggest_int("beta", -40, 40),
        "gamma": trial.suggest_float("gamma", 0.7, 1.6),
        "use_clahe": trial.suggest_categorical("use_clahe", [False, True]),
        "clahe_clip": trial.suggest_float("clahe_clip", 1.0, 4.0),
        # Raspi4 を想定し、重い CLAHE の tileGridSize は最小限に固定
        "clahe_tile": trial.suggest_categorical("clahe_tile", [4]),
        "blur_ksize": trial.suggest_categorical("blur_ksize", [1, 3, 5]),
        "confidence": trial.suggest_float("confidence", 0.1, 0.6),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Optunaで前処理パラメータを最適化して猫検出精度を上げる"
    )
    parser.add_argument("--trials", type=int, default=30, help="試行回数")
    # --output の代わりに、どのモデルをチューニングするか指定できるように変更
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="チューニングするモデルのパス (指定しない場合は config.MODEL_PATH)",
    )
    args = parser.parse_args()

    dataset_dir = Path("dataset")
    with_cat_dir = dataset_dir / "with_cat"
    without_cat_dir = dataset_dir / "without_cat"

    if not with_cat_dir.exists() or not without_cat_dir.exists():
        raise FileNotFoundError(
            f"dataset/with_cat または dataset/without_cat が見つかりません: {dataset_dir}"
        )

    with_cat_images = load_image_paths(with_cat_dir)
    without_cat_images = load_image_paths(without_cat_dir)
    total_images = len(with_cat_images) + len(without_cat_images)

    if not with_cat_images or not without_cat_images:
        raise RuntimeError(
            "学習用画像が不足しています。with_cat と without_cat を確認してください。"
        )

    print("モデルをロードしています...")
    detector = CatDetector(model_path=args.model)
    model_stem = detector.model_path.stem  # 例: yolo26n_size320_onnx_fp32
    best_params_dir = Path("best_params")
    best_params_dir.mkdir(parents=True, exist_ok=True)
    output_path = best_params_dir / f"{model_stem}.json"

    print(f"対象モデル: {detector.model_path.name}")
    print(f"パラメータ保存先: {output_path}")

    base_params = load_best_params(output_path)

    # チューニング全体で1回だけウォームアップ（初回キャッシュ生成などを除外）
    warmup_path = (with_cat_images[0] if with_cat_images else None) or (
        without_cat_images[0] if without_cat_images else None
    )
    if warmup_path is not None:
        import cv2

        warmup_frame = cv2.imread(str(warmup_path))
        if warmup_frame is not None:
            try:
                detector.detect_cat(warmup_frame)
            except Exception:
                pass

    base_metrics = evaluate(
        detector=detector,
        with_cat_images=with_cat_images,
        without_cat_images=without_cat_images,
        params=base_params,
        warmup=False,
    )
    base_fps = (
        (total_images / base_metrics["inference_time_sec"])
        if base_metrics.get("inference_time_sec", 0.0) > 0 and total_images > 0
        else 0.0
    )
    base_metrics["fps"] = base_fps

    print("\n[Baseline]")
    print(
        f"Accuracy={base_metrics['accuracy']:.2%}, Precision={base_metrics['precision']:.2%}, "
        f"Recall={base_metrics['recall']:.2%}, F1={base_metrics['f1']:.2%}"
    )
    print(f"FPS={base_metrics['fps']:.2f}")

    def objective(trial):
        params = build_trial_params(trial)

        metrics = evaluate(
            detector=detector,
            with_cat_images=with_cat_images,
            without_cat_images=without_cat_images,
            params=params,
            warmup=False,
        )
        fps = (
            (total_images / metrics["inference_time_sec"])
            if metrics.get("inference_time_sec", 0.0) > 0 and total_images > 0
            else 0.0
        )
        metrics["fps"] = fps

        trial.set_user_attr("metrics", metrics)
        # 指標: 効率スコア = F1 * FPS
        # 目標FPSを下回る場合は強く減点して「遅すぎる設定」を排除する
        target_fps = 5.0
        score = metrics["f1"] * metrics["fps"]
        if metrics["fps"] < target_fps and target_fps > 0:
            ratio = metrics["fps"] / target_fps
            score *= ratio**4
        return score

    study = optuna.create_study(direction="maximize")
    study.enqueue_trial(base_params)
    # study.optimize(objective, n_trials=args.trials, callbacks=[stop_when_one])
    study.optimize(objective, n_trials=args.trials)

    best_metrics = study.best_trial.user_attrs["metrics"]
    best_result = {
        "best_trial": study.best_trial.number,
        "best_value_score": float(study.best_value)
        if study.best_value is not None
        else None,
        "best_value_f1": float(best_metrics.get("f1", 0.0)),
        "best_params": study.best_params,
        "best_metrics": best_metrics,
        "baseline_metrics": base_metrics,
    }

    output_path.write_text(
        json.dumps(best_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n[Best Trial]")
    best_score = study.best_value if study.best_value is not None else 0.0
    print(
        f"trial={study.best_trial.number}, Score(F1*FPS)={best_score:.4f}, "
        f"F1={best_metrics['f1']:.2%}, FPS={best_metrics['fps']:.2f}"
    )
    print(
        f"Accuracy={best_metrics['accuracy']:.2%}, Precision={best_metrics['precision']:.2%}, "
        f"Recall={best_metrics['recall']:.2%}"
    )
    print(f"best_params={study.best_params}")
    print(f"結果を保存しました: {output_path}")


if __name__ == "__main__":
    main()
