from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms


CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "best_model_unfrozen.pt"


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


class EcoSortPredictor:
    """Loads the trained EcoSort model once and predicts one image at a time."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.mobilenet_v2(weights=None)
        self.model.classifier[1] = nn.Linear(1280, len(CLASSES))
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

        self.prepare = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def predict(self, image_source: str | Path | BinaryIO, top_k: int = 3) -> list[Prediction]:
        image = Image.open(image_source)
        image = ImageOps.exif_transpose(image).convert("RGB")
        batch = self.prepare(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            probabilities = torch.softmax(self.model(batch), dim=1)[0]
            values, indices = torch.topk(
                probabilities, k=min(top_k, len(CLASSES))
            )

        return [
            Prediction(label=CLASSES[index.item()], confidence=value.item())
            for value, index in zip(values, indices)
        ]
