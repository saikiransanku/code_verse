"""FastAPI app for wheat disease prediction and advisory generation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import get_settings
from backend.llm_service import LLMAdvisoryService
from backend.model import WheatDiseaseModelService
from backend.preprocess import analyze_image_quality, load_image, preprocess_for_inference

app = FastAPI(title="Wheat Disease Advisory API", version="1.0.0")

settings = get_settings()
model_service = WheatDiseaseModelService(settings.model_path)
llm_service = LLMAdvisoryService(settings)


@app.on_event("startup")
def startup_event() -> None:
    """Attempt to warm the model on API boot."""
    model_service.load()


@app.get("/health")
def healthcheck() -> dict[str, Any]:
    """Simple backend health endpoint."""
    return {
        "status": "ok",
        "model_loaded": model_service.model is not None,
        "model_error": model_service.load_error,
        "llm_provider": settings.llm_provider,
        "llm_ready": llm_service.client is not None,
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict[str, Any]:
    """Receive a wheat leaf image, classify it, and attach LLM advice."""
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    if model_service.model is None and model_service.load_error:
        raise HTTPException(status_code=503, detail=model_service.load_error)

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    try:
        pil_image = load_image(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

    quality = analyze_image_quality(pil_image, settings)
    batch_tensor = preprocess_for_inference(pil_image)

    try:
        prediction = model_service.predict(batch_tensor)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    retake_recommended = prediction.confidence < settings.confidence_threshold
    advisory = (
        {
            "explanation": "The model is not confident enough to give reliable advice yet.",
            "treatment": ["Please retake the image before applying treatment."],
            "prevention": [
                "Capture the leaf in daylight.",
                "Keep the image sharp and fill most of the frame with the leaf.",
            ],
            "source": "system",
        }
        if retake_recommended
        else llm_service.generate(prediction.display_name, prediction.confidence)
    )

    return {
        "predicted_class": prediction.display_name,
        "raw_class": prediction.predicted_class,
        "confidence": round(prediction.confidence, 4),
        "top_predictions": prediction.top_predictions,
        "retake_recommended": retake_recommended,
        "confidence_threshold": settings.confidence_threshold,
        "quality": quality,
        "advisory": advisory,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
