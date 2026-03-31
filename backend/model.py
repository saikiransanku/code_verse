"""Model loading and prediction service for inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from model.constants import CLASS_NAMES, DISPLAY_NAMES
from model.network import build_efficientnet_b0


@dataclass
class PredictionResult:
    predicted_class: str
    display_name: str
    confidence: float
    top_predictions: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


class WheatDiseaseModelService:
    """Lazy-loading wrapper around the EfficientNetB0 checkpoint."""

    def __init__(self, checkpoint_path: Path) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: torch.nn.Module | None = None
        self.class_names = list(CLASS_NAMES)
        self.display_names = dict(DISPLAY_NAMES)
        self.load_error: str | None = None

    def load(self) -> None:
        """Load the saved model checkpoint if available."""
        if not self.checkpoint_path.exists():
            self.load_error = f"Checkpoint not found: {self.checkpoint_path}"
            return

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.class_names = checkpoint.get("class_names", self.class_names)
        self.display_names = checkpoint.get("display_names", self.display_names)

        self.model = build_efficientnet_b0(num_classes=len(self.class_names), pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.load_error = None

    def predict(self, batch_tensor: torch.Tensor) -> PredictionResult:
        """Predict class probabilities from a preprocessed image tensor."""
        if self.model is None:
            if self.load_error is None:
                self.load()
            if self.model is None:
                raise RuntimeError(self.load_error or "Model could not be loaded.")

        batch_tensor = batch_tensor.to(self.device)

        with torch.inference_mode():
            logits = self.model(batch_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        confidence, predicted_idx = torch.max(probabilities, dim=0)
        predicted_class = self.class_names[int(predicted_idx)]

        top_values, top_indices = torch.topk(probabilities, k=min(3, len(self.class_names)))
        top_predictions = [
            {
                "label": self.display_names.get(self.class_names[int(index)], self.class_names[int(index)]),
                "confidence": float(value),
            }
            for value, index in zip(top_values.cpu().tolist(), top_indices.cpu().tolist())
        ]

        return PredictionResult(
            predicted_class=predicted_class,
            display_name=self.display_names.get(predicted_class, predicted_class.title()),
            confidence=float(confidence),
            top_predictions=top_predictions,
        )
