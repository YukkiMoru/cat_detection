#!/usr/bin/env python
# uv run tools/check_benchmark_result.py
# ../benchmark_results.csv -> plot (GUI)

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_benchmark_results(csv_path):
    if not csv_path.exists():
        return

    # データ読み込みと加工を一括処理
    df = pd.read_csv(csv_path)
    df["FPS"] = df["FPS"].astype(float)
    df["F1"] = df["F1_Score"].str.rstrip("%").astype(float)
    df["Model"] = df["Model_Name"].str.split("_").str[0]
    df["Type"] = df["Model_Name"].str.split("_").str[-1].str.upper()

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(11, 7))
    palette = {"MNN": "tab:red", "ONNX": "tab:blue"}
    order = {"FP32": 0, "FP16": 1, "INT8": 2}

    # 1. 軌跡（線）: 順序をソートして描画
    for _, g in df.assign(r=df["Type"].map(order)).groupby(["Model", "Size", "Format"]):
        g = g.sort_values("r")
        plt.plot(
            g["FPS"],
            g["F1"],
            color=palette[g["Format"].iloc[0]],
            alpha=0.2,
            ls="--",
            zorder=1,
        )

    # 2. 散布図
    sns.scatterplot(
        data=df,
        x="FPS",
        y="F1",
        hue="Format",
        style="Model",
        size="Size",
        sizes=(80, 200),
        palette=palette,
        alpha=0.8,
        zorder=2,
    )

    # 3. ラベル: 上下交互に配置して重なり回避
    for i, r in df.iterrows():
        plt.text(
            r["FPS"] + 0.3,
            r["F1"] + (0.4 if i % 2 else -0.7),
            r["Type"],
            fontsize=8,
            alpha=0.7,
            bbox=dict(fc="w", ec="none", alpha=0.5),
        )

    plt.title("Benchmark Result", fontweight="bold")
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_benchmark_results(Path(__file__).parent.parent / "benchmark_results.csv")
