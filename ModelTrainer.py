import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from tqdm import tqdm
import optuna

# --- Dataset Class ---
class ScrewDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, label_name)
            if not os.path.isdir(class_dir):
                continue
            for filename in os.listdir(class_dir):
                if filename.endswith(('.jpg', '.png', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, filename))
                    self.labels.append(label_name)

        self.le = LabelEncoder()
        self.labels = self.le.fit_transform(self.labels)
        self.classes = self.le.classes_

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = cv2.imread(img_path)
        image = cv2.resize(image, (128, 128))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.uint8)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)

# --- Create Model ---
def create_model(num_classes):
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# --- Train Function (with optional validation) ---
def train(model, train_loader, val_loader, device, epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

        acc = correct / len(train_loader.dataset)
        print(f"Train Loss: {running_loss:.4f}, Train Accuracy: {acc:.4f}")

        if val_loader is not None:
            model.eval()
            correct = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
            val_acc = correct / len(val_loader.dataset)
            print(f"Validation Accuracy: {val_acc:.4f}")

    return model

# --- Evaluation Function ---
def evaluate(model, loader, device, classes):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

# --- Hyperparameter Tuning with Optuna ---
def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])

    dataset = ScrewDataset("dataset", transform=data_transform)
    kfold = KFold(n_splits=5, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    acc_scores = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        model = create_model(num_classes=len(dataset.classes))
        model = train(model, train_loader, val_loader, device, epochs=5, lr=lr)

        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc_scores.append(correct / total)

    return np.mean(acc_scores)

# --- Data Transformations ---
data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Main Script ---
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print("Best hyperparameters:", study.best_params)
    best_lr = study.best_params['lr']
    best_batch_size = study.best_params['batch_size']

    dataset = ScrewDataset("dataset", transform=data_transform)
    kfold = KFold(n_splits=5, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\nFold {fold+1}")
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=best_batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=best_batch_size, shuffle=False)

        model = create_model(num_classes=len(dataset.classes))
        model = train(model, train_loader, val_loader, device, epochs=10, lr=best_lr)
        evaluate(model, val_loader, device, dataset.classes)

    # --- Final Training on Full Dataset ---
    print("\nTraining final model on full dataset with best hyperparameters...")
    full_loader = DataLoader(dataset, batch_size=best_batch_size, shuffle=True)
    final_model = create_model(num_classes=len(dataset.classes))
    final_model = train(final_model, full_loader, val_loader=None, device=device, epochs=10, lr=best_lr)
    torch.save(final_model.state_dict(), "screw_classifier_resnet_Final.pth")
    print("Final model trained on full dataset and saved as 'screw_classifier_resnet_Final_new.pth'")
