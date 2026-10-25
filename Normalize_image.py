
from pathlib import Path
from PIL import Image
import numpy as np

# Find our dataset
project_folder = Path(__file__).resolve().parent.parent
dataset_folder = project_folder / "dataset-resized" / "dataset-resized"

# Select the first cardboard image
cardboard_folder = dataset_folder / "cardboard"
image_path = sorted(cardboard_folder.glob("*.jpg"))[0]

# Open the image
image = Image.open(image_path).convert("RGB")

# Convert image into a NumPy array
pixels = np.array(image)

# Normalize pixel values
normalized_pixels = pixels.astype("float32") / 255.0

# Display results
print("Original pixel:", pixels[0, 0])
print("Normalized pixel:", normalized_pixels[0, 0])

print("Original minimum:", pixels.min())
print("Original maximum:", pixels.max())

print("Normalized minimum:", normalized_pixels.min())
print("Normalized maximum:", normalized_pixels.max())
