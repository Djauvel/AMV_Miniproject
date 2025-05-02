import cv2
import torch
import numpy as np
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import os

# --- Load label classes ---
def get_classes_from_dataset(dataset_path="dataset"):
    label_names = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    le = LabelEncoder()
    le.fit(label_names)
    return le.classes_

# --- Define Transform (must match training) ---
data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),  # Ensure consistent resizing
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Load the trained model ---
def load_model(model_path="screw_classifier_resnet_Final.pth", num_classes=3):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# --- Main Webcam Inference Function ---
def run_webcam_inference():
    classes = get_classes_from_dataset("dataset")
    model = load_model(num_classes=len(classes))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img = data_transform(img).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(img)
            _, pred = torch.max(output, 1)
            label = classes[pred.item()]
            confidence = torch.softmax(output, dim=1)[0][pred.item()].item()

        # Display prediction
        display_text = f"Class: {label} ({confidence:.2f})"
        cv2.putText(frame, display_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Screw Classifier", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- Run ---
if __name__ == "__main__":
    run_webcam_inference()
