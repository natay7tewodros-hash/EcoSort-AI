
from pathlib import Path
from preprocess_image import preprocess_image

project_folder = Path(__file__).resolve().parent.parent
dataset_folder = project_folder / "dataset-resized" / "dataset-resized"

categories = [
    "cardboard", "glass", "metal",
    "paper", "plastic", "trash"
]

total_success = 0
total_failed = 0

for category in categories:
    folder = dataset_folder / category

    for image_path in folder.glob("*.jpg"):
        try:
            result = preprocess_image(image_path)

            if result.shape != (128, 128, 3):
                raise ValueError("Unexpected image shape")

            total_success += 1

        except Exception as error:
            total_failed += 1
            print("Failed:", image_path.name, error)

print("Successfully processed:", total_success)
print("Failed images:", total_failed)
