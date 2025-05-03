from ultralytics import YOLO
import os

# --- CONFIGURATION ---
DATA_YAML = "data.yaml"
MODEL_TYPE = "yolov8s.pt"        
IMAGE_SIZE = 640
NUM_TRIALS = 25                  
EPOCHS_PER_TRIAL = 25            
PROJECT_NAME = "yolo_screw_detector_tuned"
EXPERIMENT_NAME = "bayes_opt"

# --- BAYESIAN OPTIMIZATION ---
def tune():
    print("🧪 Starting hyperparameter tuning with Optuna...")

    model = YOLO(MODEL_TYPE)

    # Tune will try NUM_TRIALS hyperparameter combinations using Optuna
    model.tune(
        data=DATA_YAML,
        imgsz=IMAGE_SIZE,
        epochs=EPOCHS_PER_TRIAL,
        iterations=NUM_TRIALS,
        name=EXPERIMENT_NAME,
        project=PROJECT_NAME,
    )

    print("✅ Tuning complete.")
    print(f"📁 Best results and model saved in: {os.path.join(PROJECT_NAME, EXPERIMENT_NAME)}")

if __name__ == "__main__":
    tune()
