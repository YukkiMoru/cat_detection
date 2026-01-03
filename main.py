import cv2
import time
import requests
import threading
import yaml
import logging
import signal
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from ultralytics import YOLO

# ロギング設定: タイムスタンプ付きでログを表示
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class Config:
    """設定を管理するクラス"""
    def __init__(self, config_path="config.yml"):
        self.config = self._load_config(config_path)
        
        # App settings
        self.headless = self._get('app.headless', False)
        
        # Discord settings
        self.webhook_url = self._load_webhook_url()
        
        # Camera settings
        self.width = self._get('camera.width', 320)
        self.height = self._get('camera.height', 240)
        self.fps = self._get('camera.fps', 2)
        
        # Detection settings
        self.duration_threshold = self._get('detection.duration_threshold', 3.0)
        self.reset_threshold = self._get('detection.reset_threshold', 1.0)
        self.confidence = self._get('detection.confidence', 0.4)
        self.class_id = self._get('detection.class_id', 15) # 15: cat
        self.use_motion_filter = self._get('detection.use_motion_filter', True)
        self.motion_threshold = self._get('detection.motion_threshold', 500)
        self.debug_motion = self._get('detection.debug_motion', False)

    def _load_config(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Config load failed: {e}. Using defaults.")
            return {}

    def _get(self, key, default):
        """ドット区切りのキーで設定値を取得"""
        v = self.config
        for k in key.split('.'):
            if isinstance(v, dict):
                v = v.get(k)
            else:
                return default
        return v if v is not None else default

    def _load_webhook_url(self):
        path = self._get('discord.webhook_url_file', ".secrets/DWU")
        try:
            return Path(path).read_text(encoding='utf-8').strip()
        except Exception:
            logger.warning("Webhook URL file not found or unreadable. Notifications disabled.")
            return ""

class CatDetector:
    """猫検出アプリケーションのメインクラス"""
    def __init__(self, config: Config):
        self.cfg = config
        logger.info("Loading YOLO model...")
        self.model = YOLO('yolo11n.pt')
        self.running = True
        self.cap = None
        
        # Detection state
        self.start_time = None
        self.last_seen_time = 0
        self.notified = False
        self.prev_gray = None

        # Signal handling (Ctrl+C で安全に終了するため)
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

    def stop(self, signum=None, frame=None):
        logger.info("Stopping application...")
        self.running = False

    def notify(self, frame):
        """Discordに通知を送る（画像付き）"""
        if not self.cfg.webhook_url.startswith("http"):
            return

        # 画像をメモリ上でJPEGにエンコード
        success, encoded_img = cv2.imencode('.jpg', frame)
        if not success:
            logger.error("Failed to encode image for notification")
            return
        
        # マルチパート形式でファイルを準備
        files = {
            'file': ('cat.jpg', encoded_img.tobytes(), 'image/jpeg')
        }
        
        ts = datetime.now(timezone(timedelta(hours=9))).strftime('%Y/%m/%d %H:%M:%S')
        data = {
            "content": f"@everyone [{ts}] 猫検出！🐈"
        }

        def _send():
            try:
                requests.post(self.cfg.webhook_url, data=data, files=files, timeout=10)
                logger.info("Notification sent successfully!")
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")

        # メインループを止めないように別スレッドで送信
        threading.Thread(target=_send, daemon=True).start()

    def process_motion(self, frame):
        """モーション検知フィルター。動きがなければFalseを返す"""
        if not self.cfg.use_motion_filter:
            return True

        # グレースケール変換とぼかし
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return False

        # フレーム間の差分を計算
        frame_delta = cv2.absdiff(self.prev_gray, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        score = cv2.countNonZero(thresh)
        self.prev_gray = gray

        # 動きが閾値以下の場合
        if score < self.cfg.motion_threshold:
            if self.cfg.debug_motion and self.start_time is None:
                logger.debug(f"Motion skip: {score}")
            
            # すでに検出中の場合(start_timeがある)は、猫がじっとしている可能性があるので
            # YOLO推論を継続させる（Trueを返す）。検出中でなければスキップ（False）。
            return self.start_time is not None
        
        return True

    def run(self):
        logger.info("Starting camera...")
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        # カメラ設定
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)

        logger.info("Monitoring started. Press Ctrl+C to stop.")

        try:
            while self.running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue

                # 1. モーション検知チェック
                should_run_yolo = self.process_motion(frame)
                
                # 2. YOLO推論
                results = []
                if should_run_yolo:
                    # verbose=Falseでコンソール出力を抑制
                    results = self.model(frame, classes=[self.cfg.class_id], conf=self.cfg.confidence, verbose=False)

                now = time.time()
                detected = False
                
                # 3. 検出判定ロジック
                if should_run_yolo and results and results[0].boxes:
                    detected = True
                    self.last_seen_time = now
                    
                    if self.start_time is None:
                        self.start_time = now
                        logger.info("Cat detected (start)")
                    
                    # 継続時間が閾値を超え、かつ未通知の場合
                    if not self.notified and (now - self.start_time >= self.cfg.duration_threshold):
                        logger.info(f"Threshold passed ({self.cfg.duration_threshold}s). Sending notification.")
                        self.notify(frame)
                        self.notified = True
                
                # 猫が見えなくなってから一定時間経過したらリセット
                elif self.start_time and (now - self.last_seen_time > self.cfg.reset_threshold):
                    logger.info("Cat lost. Resetting state.")
                    self.start_time = None
                    self.notified = False

                # 4. 画面表示（ヘッドレスモードでなければ）
                if not self.cfg.headless:
                    # 検出時はバウンディングボックス付き、そうでなければ生のフレームを表示
                    annotated_frame = results[0].plot() if (should_run_yolo and results) else frame
                    cv2.imshow("YOLO", annotated_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    # ヘッドレス時はCPU負荷を下げるため少し待機
                    time.sleep(0.01)

        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            logger.info("Cleanup done.")

if __name__ == "__main__":
    config = Config()
    detector = CatDetector(config)
    detector.run()