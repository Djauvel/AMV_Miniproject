from ultralytics import YOLO
import os
import yaml

# --- CONFIGURATION ---
DATA_YAML = "data.yaml"
MODEL_TYPE = "yolov8s.pt"
IMAGE_SIZE = 640
EPOCHS = 100
BATCH_SIZE = 16
PROJECT_NAME = "screw_detector_retrain"
EXPERIMENT_NAME = "from_best_hyp"
BEST_HYP_PATH = "yolo_screw_detector_tuned/bayes_opt/best_hyperparameters.yaml"  # Update this path as needed

# --- TRAINING ---
def retrain_with_best_hyperparams():
    print("🔁 Retraining YOLOv8 using best hyperparameters...")

    if not os.path.exists(BEST_HYP_PATH):
        print(f"❌ Could not find hyp.yaml at: {BEST_HYP_PATH}")
        return

    # Load best hyperparameters from YAML
    with open(BEST_HYP_PATH, "r") as f:
        best_hyp = yaml.safe_load(f)

    model = YOLO(MODEL_TYPE)

    # Unpack hyp dict into kwargs
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        name=EXPERIMENT_NAME,
        project=PROJECT_NAME,
        patience=10,
        **best_hyp
    )

    print(f"✅ Retraining complete. Results in: {os.path.join(PROJECT_NAME, EXPERIMENT_NAME)}")

if __name__ == "__main__":
    retrain_with_best_hyperparams()
