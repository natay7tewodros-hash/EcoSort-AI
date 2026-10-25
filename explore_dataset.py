
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Find our dataset
project_folder = Path(__file__).resolve().parent.parent
dataset_folder = project_folder / "dataset-resized" / "dataset-resized"

# Our six waste categories
categories = [
    "cardboard", "glass", "metal",
    "paper", "plastic", "trash"
]

# Create a figure with 2 rows and 3 columns
fig, axes = plt.subplots(2, 3, figsize=(10, 6))

# Visit every waste category
for i, category in enumerate(categories):

    folder = dataset_folder / category

    # Select the first image
    image_path = sorted(folder.glob("*.jpg"))[0]
    image = Image.open(image_path)

    # Find the correct position
    row = i // 3
    col = i % 3

    # Display the image
    axes[row, col].imshow(image)
    axes[row, col].set_title(category)
    axes[row, col].axis("off")

plt.tight_layout()
plt.show()
