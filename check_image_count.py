
from pathlib import Path

project = Path(__file__).resolve().parent.parent

dataset = project / "dataset-resized" / "dataset-resized"

total = 0

for folder in sorted(dataset.iterdir()):
    if folder.is_dir():
        images = list(folder.glob("*.jpg"))
        count = len(images)

        print(folder.name, ":", count)
        total += count

print("Total:", total)



