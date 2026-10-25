
from pathlib import Path
from PIL import Image
import numpy as np

# Find our cardboard images
project_folder = Path(__file__).resolve().parent.parent
dataset_folder = project_folder / "dataset-resized" / "dataset-resized"
cardboard_folder = dataset_folder / "cardboard"

# Select the first image
image_path = sorted(cardboard_folder.glob("*.jpg"))[0]

# Open image and convert to RGB
image = Image.open(image_path).convert("RGB")

# Convert the image into numbers
pixels = np.array(image)

print("Image size:", image.size)
print("Array shape:", pixels.shape)
print("First pixel RGB values:", pixels[0, 0])
print("Minimum pixel value:", pixels.min())
print("Maximum pixel value:", pixels.max())

