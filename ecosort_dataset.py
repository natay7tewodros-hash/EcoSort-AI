from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

PROJECT = Path(r"C:\Users\Admin - PC\Desktop\EcoSort AI Project")
CSV_FILE = PROJECT / "dataset_split_clean.csv"

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASS_TO_NUMBER = {name: number for number, name in enumerate(CLASSES)}


class EcoSortDataset(Dataset):
    def __init__(self, split):
        table = pd.read_csv(CSV_FILE)
        self.table = table[table["split"] == split].reset_index(drop=True)
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        image = Image.open(PROJECT / row["image_path"]).convert("RGB")
        image = self.transform(image)
        label = CLASS_TO_NUMBER[row["category"]]
        return image, label


if __name__ == "__main__":
    all_rows = pd.read_csv(CSV_FILE)
    print(all_rows["split"].value_counts())
    print()
    for split_name in all_rows["split"].unique():
        data = EcoSortDataset(split_name)
        image, label = data[0]
        print(split_name, "->", len(data), "images | first image shape:",
              tuple(image.shape), "| label:", label, "=", CLASSES[label])