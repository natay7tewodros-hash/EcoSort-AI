import time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from ecosort_dataset import EcoSortDataset

torch.manual_seed(42)

PROJECT = Path(r"C:\Users\Admin - PC\Desktop\EcoSort AI Project")
SAVE_PATH = PROJECT / "best_model_transfer.pt"

EPOCHS = 8
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Pictures must be prepared the same way the pretrained model was trained
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_data = EcoSortDataset("train")
train_data.transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

val_data = EcoSortDataset("validation")
val_data.transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

# Load MobileNetV2 with its pretrained knowledge (downloads about 14 MB once)
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# Freeze everything it already knows
for parameter in model.features.parameters():
    parameter.requires_grad = False

# Replace the last layer with a new one for our 6 classes
model.classifier[1] = nn.Linear(1280, 6)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

best_val_accuracy = 0.0
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    model.features.eval()   # keep the frozen part exactly as it was
    train_loss = 0.0
    train_correct = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_correct += (outputs.argmax(dim=1) == labels).sum().item()

    model.eval()
    val_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            val_correct += (outputs.argmax(dim=1) == labels).sum().item()

    train_accuracy = train_correct / len(train_data)
    val_accuracy = val_correct / len(val_data)
    average_loss = train_loss / len(train_data)

    print(f"Epoch {epoch}/{EPOCHS} | loss {average_loss:.3f} | "
          f"train acc {train_accuracy:.1%} | validation acc {val_accuracy:.1%} | "
          f"time {time.time() - start_time:.0f}s")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), SAVE_PATH)

print("Done. Best validation accuracy:", f"{best_val_accuracy:.1%}")
print("Best model saved to:", SAVE_PATH)