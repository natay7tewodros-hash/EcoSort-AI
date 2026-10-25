
from pathlib import Path
from PIL import Image

# Find our dataset
project_folder = Path(__file__).resolve().parent.parent
dataset_folder = project_folder / "dataset-resized" / "dataset-resized"

# Select a cardboard image
cardboard_folder = dataset_folder / "cardboard"
image_path = sorted(cardboard_folder.glob("*.jpg"))[0]

# Open the original image
image = Image.open(image_path)

print("Original size:", image.size)

# Resize the image
resized_image = image.resize((128, 128))

print("Resized size:", resized_image.size)

# Display both images
image.show()
resized_image.show()
