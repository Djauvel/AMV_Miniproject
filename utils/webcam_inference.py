import cv2
import torch
from torchvision import transforms
from models.cnn_model import ScrewClassifierCNN

def run_live_inference(model_path, class_names, image_size=224):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ScrewClassifierCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = transforms.ToPILImage()(image)
        input_tensor = transform(image_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, pred = torch.max(outputs, 1)
            label = class_names[pred.item()]

        cv2.putText(frame, f"Predicted: {label}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Screw Classifier", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
