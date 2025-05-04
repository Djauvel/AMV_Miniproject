from flask import Flask, Response, render_template_string
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import ResNet18_Weights
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import os

# --- CONFIGURATION ---
MODEL_PATH = "screw_classifier_resnet_Final.pth"
DATASET_PATH = "datasets/Olddataset"  # Folder containing training subfolders
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.5

# --- Load labels dynamically using LabelEncoder ---
def get_classes(dataset_path):
    label_names = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    le = LabelEncoder()
    le.fit(label_names)
    return list(le.classes_)

# --- Load model ---
def load_model(model_path, num_classes):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

# --- Image preprocessing (must match training) ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load model and class names once
CLASSES = get_classes(DATASET_PATH)
model = load_model(MODEL_PATH, num_classes=len(CLASSES))

# --- Flask App ---
app = Flask(__name__)

html_template = """
<!doctype html>
<title>Screw Classifier - Webcam</title>
<h2 style="text-align:center;">Live ResNet Classification</h2>
<img src="{{ url_for('video_feed') }}" width="640">
"""

# --- Frame classifier ---
def classify_frame(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img_rgb).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)[0]
        conf, pred = torch.max(probs, 0)

    label = CLASSES[pred.item()]
    return label, conf.item()

# --- Frame generator ---
def generate_frames():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    while True:
        success, frame = cap.read()
        if not success:
            break

        label, conf = classify_frame(frame)
        text = f"{label} ({conf*100:.1f}%)" if conf > CONFIDENCE_THRESHOLD else "Unknown"

        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# --- Routes ---
@app.route('/')
def index():
    return render_template_string(html_template)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Run server ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
