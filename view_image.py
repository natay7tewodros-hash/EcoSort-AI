
from pathlib import Path
from PIL import Image

# Find our dataset
project_folder = Path(__file__).resolve().parent.parent
dataset_folder = project_folder / "dataset-resized" / "dataset-resized"

# Find the cardboard images
cardboard_folder = dataset_folder / ("paper")

# Get the first JPG image
image_files = sorted(cardboard_folder.glob("*.jpg"))
image_path = image_files[0]

# Open the image
image = Image.open(image_path)

# Display information
print("Image name:", image_path.name)
print("Image size:", image.size)

# Show the image
image.show()
