from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import confusion_matrix, classification_report
from ecosort_dataset import EcoSortDataset, CLASSES

PROJECT = Path(r"C:\Users\Admin - PC\Desktop\EcoSort AI Project")

# Build the same model shape, then load our saved numbers into it
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(1280, 6)
model.load_state_dict(torch.load(PROJECT / "best_model_transfer.pt"))
model.eval()

# Prepare validation pictures exactly as during training
val_data = EcoSortDataset("validation")
val_data.transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
loader = DataLoader(val_data, batch_size=32)

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