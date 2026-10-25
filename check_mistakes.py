from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from ecosort_dataset import EcoSortDataset, CLASSES
from ecosort_model import SimpleCNN

PROJECT = Path(r"C:\Users\Admin - PC\Desktop\EcoSort AI Project")

model = SimpleCNN()
model.load_state_dict(torch.load(PROJECT / "best_model.pt"))
model.eval()

loader = DataLoader(EcoSortDataset("validation"), batch_size=32)

all_true = []
all_guess = []
with torch.no_grad():
    for images, labels in loader:
        outputs = model(images)
        all_guess.extend(outputs.argmax(dim=1).tolist())
        all_true.extend(labels.tolist())

print("Class order:", CLASSES)
print()
print(confusion_matrix(all_true, all_guess))
print()
print(classification_report(all_true, all_guess, target_names=CLASSES, digits=3))