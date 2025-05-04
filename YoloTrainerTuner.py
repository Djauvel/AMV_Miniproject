from ultralytics import YOLO
import os

# --- CONFIGURATION PARAMETERS ---
DATA_YAML = "data.yaml"                   # Path to data config file (class names, dataset paths)
MODEL_TYPE = "yolov8s.pt"                 # Base model: small variant of YOLOv8
IMAGE_SIZE = 640                          # Input image size (must match model's expected size)
NUM_TRIALS = 25                           # Number of Optuna trials (distinct hyperparameter sets)
EPOCHS_PER_TRIAL = 25                     # Training epochs per trial
PROJECT_NAME = "yolo_screw_detector_tuned"  # Root folder for saving tuning results
EXPERIMENT_NAME = "bayes_opt"            # Subfolder name for this specific experiment

# --- HYPERPARAMETER TUNING FUNCTION ---
def tune():
    print("🧪 Starting hyperparameter tuning with Optuna...")

    # Load the YOLOv8 model architecture and pretrained weights
    model = YOLO(MODEL_TYPE)

    # Launch Optuna-based hyperparameter tuning
    model.tune(
        data=DATA_YAML,                   # Path to dataset and class info
        imgsz=IMAGE_SIZE,                 # Resize input images to this resolution
        epochs=EPOCHS_PER_TRIAL,          # Train each trial for this many epochs
        iterations=NUM_TRIALS,            # Total number of hyperparameter trials to run
        name=EXPERIMENT_NAME,             # Experiment folder name
        project=PROJECT_NAME,             # Top-level project directory
    )

    print("✅ Tuning complete.")
    print(f"📁 Best results and model saved in: {os.path.join(PROJECT_NAME, EXPERIMENT_NAME)}")

# --- MAIN SCRIPT ENTRY POINT ---
if __name__ == "__main__":
    tune()
