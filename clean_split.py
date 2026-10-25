
from pathlib import Path
import csv

project_folder = Path(__file__).resolve().parent.parent

input_file = project_folder / "dataset_split.csv"
output_file = project_folder / "dataset_split_clean.csv"

excluded_files = {"glass176.jpg", "metal91.jpg"}

removed = 0
kept = 0

with open(input_file, "r", encoding="utf-8", newline="") as source:
    reader = csv.DictReader(source)

    with open(output_file, "w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            filename = Path(row["image_path"]).name

            if filename in excluded_files:
                print("Excluded:", filename)
                removed += 1
            else:
                writer.writerow(row)
                kept += 1

print("Images kept:", kept)
print("Records excluded:", removed)
print("Clean split saved to:", output_file)
