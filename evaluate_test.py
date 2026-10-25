from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix
from ecosort_dataset import EcoSortDataset, CLASSES

PROJECT = Path(r"C:\Users\Admin - PC\Desktop\EcoSort AI Project")

# Load the final model
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(1280, 6)
model.load_state_dict(torch.load(PROJECT / "best_model_unfrozen.pt"))
model.eval()

# Prepare the TEST pictures the same way as validation (no random flips)
test_data = EcoSortDataset("test")
test_data.transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
loader = DataLoader(test_data, batch_size=32)

all_true = []
all_guess = []
with torch.no_grad():
    for images, labels in loader:
        all_guess.extend(model(images).argmax(dim=1).tolist())
        all_true.extend(labels.tolist())

correct = sum(t == g for t, g in zip(all_true, all_guess))
print("Test images:", len(test_data))
print(f"Test accuracy: {correct} of {len(test_data)} = {correct / len(test_data):.1%}")
print()
print("Class order:", CLASSES)
print(confusion_matrix(all_true, all_guess))
print()
print(classification_report(all_true, all_guess, target_names=CLASSES, digits=3))