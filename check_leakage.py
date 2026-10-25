
from pathlib import Path
import csv
import hashlib

# Find our project and saved CSV
project_folder = Path(__file__).resolve().parent.parent
csv_file = project_folder / "dataset_split_clean.csv"

# Store each fingerprint and its split
seen_hashes = {}

duplicates = 0
checked = 0

# Read our saved dataset split
with open(csv_file, "r", encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        image_path = project_folder / row["image_path"]
        split = row["split"]

        # Read image file and calculate its fingerprint
        image_bytes = image_path.read_bytes()
        fingerprint = hashlib.sha256(image_bytes).hexdigest()

        # Check if fingerprint appeared in another split

        if fingerprint in seen_hashes:
            previous_path, previous_split = seen_hashes[fingerprint]

            if previous_split != split:
                duplicates += 1

                print("\nDuplicate pair found!")
                print("First image:", previous_path.name)
                print("First split:", previous_split)
                print("Second image:", image_path.name)
                print("Second split:", split)

        else:
            seen_hashes[fingerprint] = (image_path, split)

        checked += 1

print("Images checked:", checked)
print("Cross-split duplicate occurrences:", duplicates)
