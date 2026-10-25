import csv
from pathlib import Path
from collections import Counter

PROJECT = Path(r"C:\Users\Admin - PC\Desktop\EcoSort AI Project")
IMAGES = PROJECT / "dataset-resized" / "dataset-resized"
CSV_FILE = PROJECT / "dataset_split_clean.csv"

# 1. Every .jpg in the real picture folder
folder_files = set()
for p in IMAGES.rglob("*"):
    if p.is_file() and p.suffix.lower() == ".jpg":
        folder_files.add(str(p.relative_to(PROJECT)).lower())

# 2. Every image path listed in the CSV
csv_list = []
with open(CSV_FILE, newline="") as f:
    for row in csv.DictReader(f):
        csv_list.append(str(Path(row["image_path"])).lower())
csv_set = set(csv_list)

# 3. Totals
print("JPGs in real folder:", len(folder_files))
print("Rows in CSV:", len(csv_list))
print("Unique paths in CSV:", len(csv_set))

# 4. Counts per class
print("\nFolder, per class:", dict(Counter(Path(p).parent.name for p in folder_files)))
print("CSV, per class:   ", dict(Counter(Path(p).parent.name for p in csv_set)))

# 5. Differences
print("\nOnly in folder (not in CSV):")
for p in sorted(folder_files - csv_set):
    print("  ", p)

print("\nOnly in CSV (not in folder):")
for p in sorted(csv_set - folder_files):
    print("  ", p)

# 6. All JPGs under dataset-resized, including the __MACOSX folder
all_jpgs = [p for p in (PROJECT / "dataset-resized").rglob("*")
            if p.is_file() and p.suffix.lower() == ".jpg"]
print("\nAll JPGs under dataset-resized (incl. __MACOSX):", len(all_jpgs))