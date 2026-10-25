import importlib.util
import platform

print("Python version:", platform.python_version())

for name in ["numpy", "pandas", "PIL", "sklearn", "matplotlib", "tensorflow", "torch", "torchvision"]:
    found = importlib.util.find_spec(name) is not None
    print(name, "->", "installed" if found else "NOT installed")
