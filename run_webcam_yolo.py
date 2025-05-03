import cv2
from ultralytics import YOLO
import time

# --- CONFIG ---
#MODEL_PATH = "yolo_screw_detector_tuned/bayes_opt/weights/best.pt"  # Path to trained YOLOv8 model
MODEL_PATH = "screw_detector_retrain/from_best_hyp/weights/best.pt"  # Path to your trained YOLOv8 model

CONFIDENCE_THRESHOLD = 0.4
CAMERA_INDEX = 0  # 0 is usually the default webcam

# Load YOLOv8 model
model = YOLO(MODEL_PATH)

# Open webcam
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("✅ Webcam feed started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Run YOLO inference on the frame
    results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

    # Draw results on the frame
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("👋 Webcam closed.")
