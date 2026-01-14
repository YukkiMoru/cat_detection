# AI 猫検知システム (Cat Detection System)

PCに接続されたWebカメラの映像を解析し、猫🐈を検知したらDiscordに画像を通知するプログラムです。
YOLO (You Only Look Once) というAIモデルを使用しています。

## 📌 必要なもの

- Windows PC
- Webカメラ
- Python 3.13以上
- `uv` (高速なPythonパッケージ管理ツール)

---

## 🚀 導入手順 (Windows初心者向け)

### 1. `uv` のインストール

WindowsのPowerShellを開き、以下のコマンドをコピー＆ペーストして実行してください。これはPython環境を簡単に管理するためのツールです。

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

https://docs.astral.sh/uv/

インストールが完了したら、一度PowerShellを閉じて、開き直してください（パスを通すため）。

### 2. Python環境のセットアップ

このフォルダ（`cat_detection`）の中でPowerShellを開き、以下のコマンドを実行して必要なライブラリをインストールします。

```powershell
uv sync
```
これで、AIや画像処理に必要なプログラムが自動的にインストールされます。

---

## 🛠️ 事前準備

### 1. カメラの確認

PCに接続されているカメラが認識されているか確認します。

```powershell
uv run check_camera.py
```

実行すると、利用可能なカメラID（`0` や `1` など）が表示されます。  
基本的に `0` 番が使われますが、もし変更したい場合は `main.py` の `CAMERA_ID` を書き換えてください。

### 2. AIモデルのダウンロード

使用するAIモデルファイルをダウンロードします。

```powershell
uv run download_pts.py
```

このコマンドを実行すると `yolo26n.pt` などのモデルファイルがダウンロードされます。

### 3. 通知設定 (Discord)

猫を見つけたときに通知を送るための設定を行います。

1. プロジェクトフォルダの中に `.secrets` という新しいフォルダを作ります。
2. その中に `DWU` という名前のファイル（拡張子なし）を作ります。
3. `DWU` ファイルの中に、Discordの **Webhook URL** を貼り付けて保存してください。

※ Webhook URLの設定方法がわからない場合は、「Discord Webhook 作成」などで検索してください。

---

## ▶️ 使い方

準備ができたら、いよいよ監視システムを起動します。

```powershell
uv run main.py
```

### 操作方法
- **終了**: 画面が表示されている状態でキーボードの `q` を押すか、PowerShellで `Ctrl + C` を押すと終了します。

---

## ⚙️ 詳細仕様・設定

`main.py` ファイルの上部にある変数を変更することで、動作をカスタマイズできます。

| 変数名 | 説明 | デフォルト値 |
| :--- | :--- | :--- |
| `HEADLESS` | 画面を表示しない場合は `True` | `False` |
| `LOW_POWER` | 省電力モード（画像を縮小して処理） | `True` |
| `CAMERA_ID` | 使用するカメラの番号 | `0` |
| `FPS` | 1秒間の処理回数（少ないほど負荷が軽い） | `1` |
| `MODEL_PATH` | 使用するAIモデルファイル | `'yolo26n.pt'` |
| `CONFIDENCE` | AIの確信度（高いほど誤検知が減るが、見逃しも増える） | `0.4` (40%) |
| `MOTION_THRESH` | 動体検知の感度（低いほど少しの動きで反応） | `500` |
| `DURATION_THRESH`| 猫を検知し始めてから通知するまでの秒数 | `3.0` |
| `RESET_THRESH` | 猫を見失ってからリセットするまでの秒数 | `1.0` |

### 動作ロジック

1. **モーション検知**: まず映像に「動き」があるかをチェックします（省電力のため）。
2. **AI検知**: 動きがあった場合のみ、YOLO AIを使って「猫」がいるか判定します。
3. **継続確認**: `DURATION_THRESH` (3秒) 以上連続して猫が写り続けたら、写真を撮ってDiscordに送信します。
4. **リセット**: 猫がいなくなって `RESET_THRESH` (1秒) 経過すると、また次の猫待ち状態に戻ります。
