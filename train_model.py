import time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from ecosort_dataset import EcoSortDataset
from ecosort_model import SimpleCNN

torch.manual_seed(42)

PROJECT = Path(r"C:\Users\Admin - PC\Desktop\EcoSort AI Project")
SAVE_PATH = PROJECT / "best_model.pt"

EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

train_loader = DataLoader(EcoSortDataset("train"), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(EcoSortDataset("validation"), batch_size=BATCH_SIZE)

model = SimpleCNN()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_accuracy = 0.0
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    # ---- Learning part ----
    model.train()
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

    # ---- Checking part (no learning) ----
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            val_correct += (outputs.argmax(dim=1) == labels).sum().item()

    train_accuracy = train_correct / len(train_loader.dataset)
    val_accuracy = val_correct / len(val_loader.dataset)
    average_loss = train_loss / len(train_loader.dataset)

    print(f"Epoch {epoch}/{EPOCHS} | loss {average_loss:.3f} | "
          f"train acc {train_accuracy:.1%} | validation acc {val_accuracy:.1%} | "
          f"time {time.time() - start_time:.0f}s")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), SAVE_PATH)

print("Done. Best validation accuracy:", f"{best_val_accuracy:.1%}")
print("Best model saved to:", SAVE_PATH)