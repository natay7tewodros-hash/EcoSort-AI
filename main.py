
from pathlib import Path

# Find the main project folder
project_folder = Path(__file__).resolve().parent.parent

# Find the actual TrashNet dataset
dataset_folder = project_folder / "dataset-resized" / "dataset-resized"

# Our six waste categories
categories = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

total_images = 0

print("EcoSort AI - TrashNet Dataset")
print("-----------------------------")

# Count images in each category
for category in categories:
    folder = dataset_folder / category

    if folder.is_dir():
        image_count = len(list(folder.glob("*.jpg")))
        total_images += image_count

        print(category, ":", image_count, "images")
    else:
        print(category, ": folder not found!")

print("-----------------------------")
print("Total images:", total_images)
