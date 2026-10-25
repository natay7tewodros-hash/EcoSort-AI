from pathlib import Path
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image, ImageOps

PROJECT = Path(r"C:\Users\Admin - PC\Desktop\EcoSort AI Project")
IMAGE_FILE = PROJECT / "my_image.jpg"

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Stop with a clear message if the photo is missing
if not IMAGE_FILE.exists():
    print("Photo not found:", IMAGE_FILE)
    raise SystemExit

# Load the final model
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(1280, 6)
model.load_state_dict(torch.load(PROJECT / "best_model_unfrozen.pt"))
model.eval()

# Prepare the picture exactly like the test pictures
prepare = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

picture = Image.open(IMAGE_FILE)
picture = ImageOps.exif_transpose(picture).convert("RGB")
batch = prepare(picture).unsqueeze(0)

with torch.no_grad():
    scores = model(batch)
    chances = torch.softmax(scores, dim=1)[0]

top = torch.topk(chances, 3)
print("Picture:", IMAGE_FILE.name)
for chance, number in zip(top.values, top.indices):
    print(f"  {CLASSES[number]:<10} {chance.item():.1%}")