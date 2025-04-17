import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from torchvision import transforms

# --- Dataset Class ---
class ScrewDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Collect all images and their labels
        for label_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, label_name)
            if not os.path.isdir(class_dir):
                continue
            for filename in os.listdir(class_dir):
                if filename.endswith(('.jpg', '.png', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, filename))
                    self.labels.append(label_name)

        # Encode class names into numbers
        self.le = LabelEncoder()
        self.labels = self.le.fit_transform(self.labels)
        self.classes = self.le.classes_

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Read using OpenCV (BGR)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (128, 128))  # Resize for consistency
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to float and normalize
        image = image.astype(np.float32) / 255.0

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)

        # Convert to torch tensor and permute to CHW
        image = torch.tensor(image).permute(2, 0, 1)

        return image, torch.tensor(label)

# --- CNN Model ---
class ScrewCNN(nn.Module):
    def __init__(self, num_classes):
        super(ScrewCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 64x64
        x = self.pool(F.relu(self.conv2(x)))  # 32x32
        x = self.pool(F.relu(self.conv3(x)))  # 16x16
        x = x.view(-1, 64 * 16 * 16)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# --- Training Setup ---
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

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

        # Validation
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

# --- Main Script ---
if __name__ == "__main__":
    dataset = ScrewDataset("examroomdataset")

    # Split into training and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    num_classes = len(dataset.classes)
    model = ScrewCNN(num_classes=num_classes)

    print(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)}")
    trained_model = train_model(model, train_loader, val_loader, epochs=15, lr=0.001)

    # Save the model
    torch.save(trained_model.state_dict(), "screw_classifier.pth")
    print("Model saved as screw_classifier.pth")