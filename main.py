import os
import cv2
import torch
import numpy as np
from ModelTrainer import ScrewCNN  # assuming your model class is in model.py
from ModelTrainer import ScrewDataset  # or import it directly if in the same file
from torch.nn.functional import softmax
from torchvision import transforms
import random

# -------- Load Model --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset to get class labels
dataset = ScrewDataset("dataset")
num_classes = len(dataset.classes)
class_names = dataset.classes

# Initialize model and load weights
model = ScrewCNN(num_classes)
model.load_state_dict(torch.load("screw_classifier.pth", map_location=device))
model.to(device)
model.eval()

# -------- Prediction + Display Function --------
def predict_and_show(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (128, 128))
    image_tensor = torch.tensor(image_resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)
        label = class_names[pred.item()]
        certainty = conf.item()

    # Overlay prediction
    display_img = cv2.resize(image, (512, 512))
    text = f"{label} ({certainty * 100:.1f}%)"
    cv2.putText(display_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Show image
    cv2.imshow("Prediction", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -------- Main Script: Pick a Few Samples from Dataset --------
if __name__ == "__main__":
    sample_indices = random.sample(range(len(dataset)), 10)

    for idx in sample_indices:
        image_path = dataset.image_paths[idx]
        predict_and_show(image_path)
