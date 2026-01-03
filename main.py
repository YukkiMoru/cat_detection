import cv2
from ultralytics import YOLO

model = YOLO('yolo11n.pt')

# カメラの起動
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# COCOデータセットの猫のIDは変わらず 15 です
CAT_CLASS_ID = 15

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 推論実行（書き方はv8と同じです）
    results = model(frame, classes=[CAT_CLASS_ID], verbose=False)

    # 結果の描画
    annotated_frame = results[0].plot()

    if len(results[0].boxes) > 0:
        print("猫ちゃん検出！(YOLO11) 😺")

    cv2.imshow("YOLO11 Cat Detector", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()