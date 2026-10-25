
from pathlib import Path
from sklearn.model_selection import train_test_split

# Locate the dataset
project_folder = Path(__file__).resolve().parent.parent
dataset_folder = project_folder / "dataset-resized" / "dataset-resized"

categories = [
    "cardboard", "glass", "metal",
    "paper", "plastic", "trash"
]

image_paths = []
labels = []

# Collect image paths and their category labels
for category in categories:
    folder = dataset_folder / category

    for image_path in sorted(folder.glob("*.jpg")):
        image_paths.append(image_path)
        labels.append(category)

# Split into 70% training and 30% temporary data
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.30,
    random_state=42,
    stratify=labels
)

# Divide temporary data equally into validation and test
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.50,
    random_state=42,
    stratify=temp_labels
)

print("Training images:", len(train_paths))
print("Validation images:", len(val_paths))
print("Testing images:", len(test_paths))

print("Total images:",
      len(train_paths) + len(val_paths) + len(test_paths))

from collections import Counter

print("\nCategory distribution:")

for category in categories:
    train_count = Counter(train_labels)[category]
    val_count = Counter(val_labels)[category]
    test_count = Counter(test_labels)[category]

    print(
        category,
        "| Train:", train_count,
        "| Validation:", val_count,
        "| Test:", test_count
    )


import csv

output_file = project_folder / "dataset_split.csv"

with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow(["image_path", "category", "split"])

    for paths, labels, split_name in [
        (train_paths, train_labels, "train"),
        (val_paths, val_labels, "validation"),
        (test_paths, test_labels, "test")
    ]:
        for path, label in zip(paths, labels):
            writer.writerow([
                str(path.relative_to(project_folder)),
                label,
                split_name
            ])

print("Dataset split saved to:", output_file)
