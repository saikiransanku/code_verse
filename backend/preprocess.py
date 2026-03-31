"""Inference preprocessing and image quality checks."""

from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from backend.config import Settings
from model.constants import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE

INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def load_image(image_bytes: bytes) -> Image.Image:
    """Load a user-provided image without altering disease texture."""
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return image


def preprocess_for_inference(image: Image.Image) -> torch.Tensor:
    """Minimal inference preprocessing: resize + normalize only."""
    tensor = INFERENCE_TRANSFORM(image)
    return tensor.unsqueeze(0)


def analyze_image_quality(image: Image.Image, settings: Settings) -> dict:
    """Generate farmer-facing warnings for dark or blurry images."""
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness_score = float(gray.mean())

    warnings: list[str] = []
    if brightness_score < settings.brightness_threshold:
        warnings.append("Image looks too dark. Please retake it in better light if possible.")
    if blur_score < settings.blur_threshold:
        warnings.append("Image looks blurry. Please keep the leaf in focus and retake the photo.")

    return {
        "blur_score": blur_score,
        "brightness_score": brightness_score,
        "warnings": warnings,
    }

