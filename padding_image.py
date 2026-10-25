
from PIL import Image

# Create a blank white square, 128 pixels wide and tall
canvas = Image.new("RGB", (128, 128), "white")

# Display the square
canvas.show()

# Print its dimensions
print("Canvas size:", canvas.size)

from pathlib import Path
from PIL import Image, ImageOps

# Find the cardboard image
project_folder = Path(__file__).resolve().parent.parent
dataset_folder = project_folder / "dataset-resized" / "dataset-resized"
cardboard_folder = dataset_folder / "cardboard"

image_path = sorted(cardboard_folder.glob("*.jpg"))[0]

# Open the original image
image = Image.open(image_path).convert("RGB")

# Resize without stretching
image.thumbnail((128, 128))

# Create a white square
canvas = Image.new("RGB", (128, 128), "white")

# Calculate where to place the image
x = (128 - image.width) // 2
y = (128 - image.height) // 2

# Place the image in the center
canvas.paste(image, (x, y))

# Display the result
canvas.show()

print("Resized image size:", image.size)
print("Final canvas size:", canvas.size)
