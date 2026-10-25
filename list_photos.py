from pathlib import Path

PROJECT = Path(r"C:\Users\Admin - PC\Desktop\EcoSort AI Project")

print("Looking in:", PROJECT)
found = False
for file in PROJECT.iterdir():
    if file.is_file() and file.suffix.lower() in [".jpg", ".jpeg", ".png", ".heic", ".webp"]:
        print("Found picture:", repr(file.name))
        found = True

if not found:
    print("No pictures found directly in this folder.")
