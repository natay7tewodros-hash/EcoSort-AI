from pathlib import Path
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image, ImageOps

PROJECT = Path(r"C:\Users\Admin - PC\Desktop\EcoSort AI Project")
PHOTO_FOLDER = PROJECT / "test_photos"
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
UNSURE_BELOW = 0.60   # flag answers below 60% as unsure

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(1280, 6)
model.load_state_dict(torch.load(PROJECT / "best_model_unfrozen.pt"))
model.eval()

prepare = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

photos = sorted(p for p in PHOTO_FOLDER.iterdir()
                if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"])
if not photos:
    print("No photos found in:", PHOTO_FOLDER)
    raise SystemExit

correct = 0
scored = 0
for photo in photos:
    picture = ImageOps.exif_transpose(Image.open(photo)).convert("RGB")
    batch = prepare(picture).unsqueeze(0)
    with torch.no_grad():
        chances = torch.softmax(model(batch), dim=1)[0]
    best = chances.argmax().item()
    guess = CLASSES[best]
    confidence = chances[best].item()

    true_class = photo.stem.split("_")[0].lower()
    note = ""
    if true_class in CLASSES:
        scored += 1
        if guess == true_class:
            correct += 1
            note = "correct"
        else:
            note = f"WRONG (real: {true_class})"
    flag = "  <- UNSURE" if confidence < UNSURE_BELOW else ""
    print(f"{photo.name:<28} guess: {guess:<10} {confidence:.0%}  {note}{flag}")

if scored:
    print(f"\nCorrect: {correct} of {scored} = {correct / scored:.0%}")
print("Small informal test: not a replacement for the 89.2% test score.")