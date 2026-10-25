from torch import nn
from torch.utils.data import DataLoader
from ecosort_dataset import EcoSortDataset


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


if __name__ == "__main__":
    model = SimpleCNN()
    loader = DataLoader(EcoSortDataset("train"), batch_size=32, shuffle=True)

    images, labels = next(iter(loader))
    print("Batch of images:", images.shape)
    print("Batch of labels:", labels.shape)

    outputs = model(images)
    print("Model output:", outputs.shape)

    total = sum(p.numel() for p in model.parameters())
    print("Learnable numbers in the model:", total)