
from pathlib import Path
from preprocess_image import preprocess_image

project_folder = Path(__file__).resolve().parent.parent
dataset_folder = project_folder / "dataset-resized" / "dataset-resized"

categories = [
    "cardboard", "glass", "metal",
    "paper", "plastic", "trash"
]

for category in categories:
    folder = dataset_folder / category
    image_path = sorted(folder.glob("*.jpg"))[0]

    result = preprocess_image(image_path)

    print(category, ":", result.shape, result.dtype)
