from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ecosort_dataset import EcoSortDataset, CLASSES

PROJECT = Path(r"C:\Users\Admin - PC\Desktop\EcoSort AI Project")
OUTPUT_FILE = Path(__file__).parent / "wrong_pics.png"

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(1280, 6)
model.load_state_dict(torch.load(PROJECT / "best_model_transfer.pt"))
model.eval()

val_data = EcoSortDataset("validation")
val_data.transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
loader = DataLoader(val_data, batch_size=32)

# Find every picture the model got wrong
wrong = []
position = 0
with torch.no_grad():
    for images, labels in loader:
        guesses = model(images).argmax(dim=1)
        for true_label, guess in zip(labels.tolist(), guesses.tolist()):
            if true_label != guess:
                wrong.append((position, true_label, guess))
            position += 1

print("Total wrong:", len(wrong), "out of", len(val_data))

# Show the weakest classes first (trash, plastic, ...)
wrong.sort(key=lambda item: item[1], reverse=True)
shown = wrong[:24]

fig, axes = plt.subplots(4, 6, figsize=(14, 10))
for ax in axes.flatten():
    ax.axis("off")

for ax, (index, true_label, guess) in zip(axes.flatten(), shown):
    picture = Image.open(PROJECT / val_data.table.iloc[index]["image_path"])
    ax.imshow(picture)
    ax.set_title(f"real: {CLASSES[true_label]}\nguess: {CLASSES[guess]}", fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=100)
print("Saved picture to:", OUTPUT_FILE)