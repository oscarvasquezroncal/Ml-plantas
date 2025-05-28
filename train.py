import torch
import torchvision
from torchvision import transforms, datasets, models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report
import numpy as np
import os

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor()
])

DATASET_DIR = "dataset"
assert os.path.exists(DATASET_DIR), f"❌ Carpeta '{DATASET_DIR}' no encontrada."

full_dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)

assert len(full_dataset.classes) == 2, "Se requieren exactamente 2 clases."
assert len(full_dataset) >= 10, " Agrega más imágenes para un entrenamiento mínimo efectivo."

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Dispositivo: {device}")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_val_acc = 0
EPOCHS = 15

for epoch in range(1, EPOCHS + 1):
    model.train()
    correct, total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    print(f"🔁 Época {epoch}: Entrenamiento Accuracy = {train_acc:.2f}")

    model.eval()
    val_correct, val_total = 0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    print(f"Epoca {epoch}: Validación Accuracy = {val_acc:.2f}")

    class_names = ['Healthy', 'Diseased'] 
    unique_true = set(y_true)
    unique_pred = set(y_pred)

    if len(unique_true) > 1 and len(unique_pred) > 1:
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    else:
        print("No se puede generar classification_report (una sola clase presente).")
        print(f"y_true: {y_true}")
        print(f"y_pred: {y_pred}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "modelo_demo.pt")

print("✅ Entrenamiento finalizado con todo y fulldebgs.")
