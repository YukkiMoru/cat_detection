import argparse
import json
from pathlib import Path

import optuna

from inference import CatDetector, load_best_params

# 先ほど作成した IoU ベースの評価関数を precision.py からインポートする想定
from precision import evaluate_with_boxes


def build_trial_params(trial):
    """Optuna の探索範囲定義"""
    return {
        "alpha": trial.suggest_float("alpha", 0.3, 1.5),
        "beta": trial.suggest_int("beta", -50, 50),
        "gamma": trial.suggest_float("gamma", 0.5, 3.0),
        "use_clahe": trial.suggest_categorical("use_clahe", [False, True]),
        "clahe_clip": trial.suggest_float("clahe_clip", 1.0, 4.0),
        "clahe_tile": trial.suggest_categorical("clahe_tile", [4]),  # 固定
        "blur_ksize": trial.suggest_categorical("blur_ksize", [1, 3, 5]),
        "confidence": trial.suggest_float("confidence", 0.01, 0.7),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Optunaで前処理パラメータを最適化(dataset_boxed対応)"
    )
    parser.add_argument("--trials", type=int, default=50, help="試行回数")
    parser.add_argument("--model", type=Path, default=None, help="モデルパス")
    parser.add_argument(
        "--dir", type=Path, default=Path("dataset_boxed/val"), help="評価用ディレクトリ"
    )
    args = parser.parse_args()

    # 1. 準備
    dataset_dir = args.dir
    if not dataset_dir.exists():
        raise FileNotFoundError(f"ディレクトリが見つかりません: {dataset_dir}")

    print("📦 モデルロード中...")
    detector = CatDetector(model_path=args.model)
    model_stem = detector.model_path.stem
    output_path = Path("best_params") / f"{model_stem}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ベースラインの読み込み
    base_params = load_best_params(output_path)

    print(f"🚀 最適化開始: {model_stem}")
    print(f"📂 使用データ: {dataset_dir}")

    # 2. 目的関数の定義
    def objective(trial):
        params = build_trial_params(trial)
        detector.set_params(params)

        # IoU ベースの評価を実行
        # evaluate_with_boxes 内で TP, FP, FN, TN, f1, avg_ms が計算される
        metrics = evaluate_with_boxes(detector, dataset_dir, verbose=False)

        # FPS の算出
        avg_ms = metrics.get("avg_ms", 0.0)
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
        metrics["fps"] = fps

        trial.set_user_attr("metrics", metrics)

        # --- スコア計算ロジック ---
        # 基本スコアは F1-Score
        f1 = metrics["f1"]

        # FPS ペナルティ (Raspi4 で 5FPS は確保したい場合)
        target_fps = 5.0
        score = f1 * fps
        if fps < target_fps:
            # 目標FPSを下回る場合、急激にスコアを落とす
            score *= (fps / target_fps) ** 4

        return score

    # 3. 最適化の実行
    study = optuna.create_study(direction="maximize")

    # 現在のベスト設定を最初の試行として登録
    study.enqueue_trial(base_params)

    study.optimize(objective, n_trials=args.trials)

    # 4. 結果の保存
    best_metrics = study.best_trial.user_attrs["metrics"]
    best_result = {
        "best_trial": study.best_trial.number,
        "best_value_score": float(study.best_value),
        "best_params": study.best_params,
        "best_metrics": best_metrics,
    }

    output_path.write_text(json.dumps(best_result, indent=2), encoding="utf-8")

    print("\n" + "=" * 40)
    print("最適化完了!")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best F1   : {best_metrics['f1']:.2%}")
    print(f"Best FPS  : {best_metrics['fps']:.2f}")
    print(f"Params    : {study.best_params}")
    print(f"保存先    : {output_path}")
    print("=" * 40)


if __name__ == "__main__":
    main()
