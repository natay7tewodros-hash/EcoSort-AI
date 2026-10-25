import csv
from pathlib import Path

# 1. The main project folder (full path, so it works from anywhere)
PROJECT = Path(r"C:\Users\Admin - PC\Desktop\EcoSort AI Project")

# 2. Search the whole project for the CSV file
matches = list(PROJECT.rglob("dataset_split_clean.csv"))
print("Found CSV files:", matches)

# 3. If we found it, open it and look at the top
if len(matches) == 0:
    print("CSV not found. Tell me and we will look for it.")
else:
    csv_path = matches[0]
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        first_row = next(reader)

    print("Columns:", header)
    print("First row:", first_row)