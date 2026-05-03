"""
WildVision — Dog Breed Detection Service (YOLO11n-cls)
Uses a fine-tuned YOLO11n classification model (DOG2.pt) for 121-breed classification.
"""

import os
from ultralytics import YOLO

# Global model (lazy-loaded)
_model = None

# Path to the YOLO classification model
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "DOG2.pt"
)


def _clean_breed_name(raw_name: str) -> str:
    """
    Convert raw class names like 'n02099601-golden_retriever' into
    human-readable names like 'Golden Retriever'.
    """
    # Remove the ImageNet synset prefix (e.g. 'n02099601-')
    if "-" in raw_name:
        raw_name = raw_name.split("-", 1)[1]
    # Replace underscores with spaces and title-case
    return raw_name.replace("_", " ").title()


def _get_model():
    """Lazy-load the YOLO classification model on first request."""
    global _model
    if _model is None:
        _model = YOLO(_MODEL_PATH, task="classify")
    return _model


def detect_animal(image_path: str) -> dict:
    """
    Run YOLO classification inference on the given image.
    Returns a dict with 'name', 'confidence', and 'all_detections'.
    """
    try:
        model = _get_model()

        # Run inference
        results = model(image_path, verbose=False)
        result = results[0]  # Single image → single result

        # result.probs contains classification probabilities
        probs = result.probs

        if probs is None:
            return {
                "name": "Unknown",
                "confidence": 0.0,
                "all_detections": [],
            }

        # Get top-5 predictions
        top5_indices = probs.top5           # List of class indices
        top5_confs = probs.top5conf.tolist()  # List of confidence values
        class_names = result.names           # {index: class_name} mapping

        detections = []
        for idx, conf in zip(top5_indices, top5_confs):
            breed_name = _clean_breed_name(class_names.get(idx, f"class_{idx}"))
            detections.append({
                "name": breed_name,
                "confidence": round(conf * 100, 1),
            })

        if detections:
            best = detections[0]
            return {
                "name": best["name"],
                "confidence": best["confidence"],
                "all_detections": detections,
            }

        return {
            "name": "Unknown",
            "confidence": 0.0,
            "all_detections": [],
        }

    except Exception as e:
        return {
            "name": "Error",
            "confidence": 0.0,
            "all_detections": [],
            "error": str(e),
        }
