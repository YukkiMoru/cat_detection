import csv
import subprocess
import sys
import time
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.columns import Columns

from inference import CatDetector
from precision import evaluate, load_images

# --- 設定 ---
MAX_WORKERS = 8  # CPUに合わせて調整 (8〜16など)
TRIALS = 30      # オプティナの試行回数

def get_best_params(model_stem):
    """保存されたJSONからベストパラメータを読み取る"""
    path = Path("best_params") / f"{model_stem}.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None

def worker_task(model_path, worker_id, status_dict, images_data, leaderboard):
    """チューニングとF1スコアの確認に特化したタスク"""
    model_name = model_path.name
    current_dir = Path(__file__).parent
    tuning_script = current_dir / "tuning.py"
    
    # --- 1. チューニング実行 ---
    status_dict[worker_id] = {"model": model_name, "state": "🔥 Tuning...", "params": "Searching..."}
    
    params_json = Path("best_params") / f"{model_path.stem}.json"
    if not params_json.exists():
        try:
            # 外部プロセスでチューニングスクリプトを実行（時間はかかります）
            subprocess.run(
                [sys.executable, str(tuning_script), "--model", str(model_path), "--trials", str(TRIALS)],
                check=True, capture_output=True
            )
        except Exception as e:
            status_dict[worker_id] = {"model": model_name, "state": "[red]❌ Error[/red]", "params": "Tuning failed"}
            return None

    # チューニング結果の読み込み
    best_p = get_best_params(model_path.stem)
    if best_p:
        # JSONに保存されたキー名に合わせて適宜変更してください
        conf = best_p.get('conf', 0.0)
        iou = best_p.get('iou', 0.0)
        p_str = f"conf:{conf:.3f} iou:{iou:.3f}"
    else:
        p_str = "Default Params"

    # --- 2. F1スコアの最終確認 ---
    status_dict[worker_id] = {"model": model_name, "state": "📊 Checking F1...", "params": p_str}
    
    with_cat, without_cat = images_data
    try:
        # ※注意: CatDetector内で best_params_dir の JSON を読み込んで
        # conf と iou を適用する仕組みになっていることが前提です
        detector = CatDetector(model_path=model_path)
        metrics = evaluate(detector, with_cat, without_cat, verbose=False)
        
        res = {
            "name": model_path.stem,
            "f1": metrics['f1'],
            "params": p_str
        }
        
        # リーダーボードに追加
        leaderboard.append(res)
        status_dict[worker_id] = {"model": model_name, "state": "[green]✅ Done[/green]", "params": p_str}
        return res
    except Exception as e:
        status_dict[worker_id] = {"model": model_name, "state": "[red]❌ Error[/red]", "params": "Eval failed"}
        return None

def generate_leaderboard_table(leaderboard):
    """チューニング特化のリーダーボード（FPS排除）"""
    table = Table(title="🏆 Top 5 Models (Highest F1 Score)", expand=True, border_style="magenta")
    table.add_column("Rank", justify="center", style="bold")
    table.add_column("Model Name", style="cyan")
    table.add_column("F1 Score", justify="right", style="bold green")
    table.add_column("Best Params (conf, iou)", style="dim")

    # F1スコア順にソート
    sorted_list = sorted(list(leaderboard), key=lambda x: x['f1'], reverse=True)[:5]
    for i, item in enumerate(sorted_list, 1):
        table.add_row(str(i), item['name'], f"{item['f1']:.2%}", item['params'])
    return table

def make_layout():
    """レイアウト作成"""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3)
    )
    layout["body"].split_row(
        Layout(name="workers", ratio=3), # ワーカーの表示領域を少し広めに
        Layout(name="leaderboard", ratio=2)
    )
    return layout

def main():
    models_dir = Path("models")
    model_files = list(models_dir.rglob("*.onnx")) + list(models_dir.rglob("*.mnn"))
    
    dataset_dir = Path("dataset")
    images_data = (load_images(dataset_dir / "with_cat"), load_images(dataset_dir / "without_cat"))

    manager = Manager()
    status_dict = manager.dict({i: {"model": "---", "state": "Waiting", "params": "---"} for i in range(MAX_WORKERS)})
    leaderboard = manager.list()
    
    layout = make_layout()
    console = Console()

    with Live(layout, refresh_per_second=4, screen=True):
        layout["header"].update(Panel("🐱 CatDetector Pro: Phase 1 - Hyperparameter Tuning", style="bold white on blue"))
        
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(worker_task, p, i % MAX_WORKERS, status_dict, images_data, leaderboard): p for i, p in enumerate(model_files)}

            while any(f.running() for f in futures):
                # 前回提案した「綺麗なグリッド表示」を採用
                grid = Table.grid(expand=True)
                grid.add_column()
                grid.add_column()
                grid.add_column()
                grid.add_column() # 4列
                
                worker_panels = []
                for i in range(MAX_WORKERS):
                    s = status_dict[i]
                    worker_panels.append(Panel(f"[bold]{s['model']}[/bold]\n{s['state']}\n[dim]{s['params']}[/dim]", title=f"Worker {i}"))
                
                for i in range(0, MAX_WORKERS, 4):
                    row_panels = worker_panels[i:i+4]
                    while len(row_panels) < 4:
                        row_panels.append("")
                    grid.add_row(*row_panels)

                layout["workers"].update(grid)
                layout["leaderboard"].update(generate_leaderboard_table(leaderboard))
                time.sleep(0.2)

    console.print("\n[bold green]🏁 Tuning Phase Completed![/bold green]")

if __name__ == "__main__":
    main()