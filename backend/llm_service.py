"""LLM-backed advisory generation with a safe local fallback."""

from __future__ import annotations

import json

from backend.config import Settings

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled by runtime fallback
    OpenAI = None


FALLBACK_ADVICE = {
    "Healthy": {
        "explanation": "The leaf looks healthy for the target wheat disease classes.",
        "treatment": ["No immediate treatment is needed right now."],
        "prevention": [
            "Keep monitoring the crop every few days.",
            "Maintain balanced irrigation and nutrition.",
        ],
    },
    "Rust": {
        "explanation": "Rust usually appears as small orange-brown pustules on the leaf.",
        "treatment": [
            "Remove severely affected leaves where practical.",
            "Use a locally approved fungicide after checking local guidance.",
        ],
        "prevention": [
            "Monitor nearby plants early.",
            "Use resistant wheat varieties when available.",
        ],
    },
    "Leaf Blight": {
        "explanation": "Leaf blight often causes brown lesions and drying patches on the leaf.",
        "treatment": [
            "Remove heavily damaged plant material if practical.",
            "Use locally recommended disease management products.",
        ],
        "prevention": [
            "Avoid keeping the crop canopy wet for long periods.",
            "Improve field sanitation between crop cycles.",
        ],
    },
    "Powdery Mildew": {
        "explanation": "Powdery mildew often looks like white powder on the leaf surface.",
        "treatment": [
            "Treat early if the spread is increasing.",
            "Use locally approved fungicide advice for mildew control.",
        ],
        "prevention": [
            "Improve airflow around the crop when possible.",
            "Watch for early white patches on nearby leaves.",
        ],
    },
    "Spot Blotch": {
        "explanation": "Spot blotch often appears as dark brown spots that spread across the leaf.",
        "treatment": [
            "Apply locally recommended disease control practices.",
            "Remove severely affected leaves where practical.",
        ],
        "prevention": [
            "Use healthy seed and resistant varieties if available.",
            "Avoid carrying infected residue into the next cycle.",
        ],
    },
}


def build_prompt(disease: str, confidence: float) -> str:
    """Prompt template for concise farmer-friendly advisory text."""
    return f"""
You are an agricultural expert.

Disease: {disease}
Confidence: {confidence:.2f}

Explain in simple language:

1. What is this disease?
2. How to treat it?
3. How to prevent it?

Keep it short and farmer-friendly.
Return valid JSON with this exact shape:
{{
  "explanation": "short explanation",
  "treatment": ["tip 1", "tip 2"],
  "prevention": ["tip 1", "tip 2"]
}}
""".strip()


class LLMAdvisoryService:
    """Generate advisory content from an LLM with a rule-based fallback."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = None

        if settings.llm_provider == "openai" and settings.openai_api_key and OpenAI is not None:
            self.client = OpenAI(api_key=settings.openai_api_key)

    def generate(self, disease: str, confidence: float) -> dict:
        """Return an explanation, treatment, and prevention payload."""
        if self.client is None:
            response = dict(FALLBACK_ADVICE.get(disease, FALLBACK_ADVICE["Healthy"]))
            response["source"] = "fallback"
            return response

        prompt = build_prompt(disease=disease, confidence=confidence)

        try:
            response = self.client.responses.create(
                model=self.settings.openai_model,
                input=prompt,
            )
            text = getattr(response, "output_text", "").strip()
            parsed = json.loads(text)
            parsed["source"] = "llm"
            return parsed
        except Exception:
            response = dict(FALLBACK_ADVICE.get(disease, FALLBACK_ADVICE["Healthy"]))
            response["source"] = "fallback"
            return response

