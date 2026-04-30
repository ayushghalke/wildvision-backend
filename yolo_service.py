"""
WildVision — YOLO Detection Service
Loads the pre-trained best.pt model and runs inference on uploaded images.
Optimized for low-memory environments (Render free tier).
"""

import os
import torch

# Reduce PyTorch memory usage
torch.set_num_threads(1)

# Global model reference (lazy-loaded)
_model = None


def _get_model():
    """Lazy-load the YOLO model on first request to save startup memory."""
    global _model
    if _model is None:
        from ultralytics import YOLO

        # Check local dir first (deployment), then parent dir (local dev)
        local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")
        parent_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "best.pt")
        model_path = local_path if os.path.exists(local_path) else parent_path

        _model = YOLO(model_path)
    return _model


def detect_animal(image_path: str) -> dict:
    """
    Run YOLO inference on the given image.
    Returns a dict with 'name', 'confidence', and 'all_detections'.
    """
    try:
        model = _get_model()
        results = model(image_path)

        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            detections = []

            for box in boxes:
                class_id = int(box.cls[0])
                class_name = results[0].names[class_id]
                confidence = float(box.conf[0])
                detections.append({
                    "name": class_name,
                    "confidence": round(confidence * 100, 1)
                })

            # Sort by confidence, highest first
            detections.sort(key=lambda x: x["confidence"], reverse=True)

            # Return the best detection + all detections
            best = detections[0]
            return {
                "name": best["name"],
                "confidence": best["confidence"],
                "all_detections": detections
            }

        return {
            "name": "Unknown",
            "confidence": 0.0,
            "all_detections": []
        }

    except Exception as e:
        return {
            "name": "Error",
            "confidence": 0.0,
            "all_detections": [],
            "error": str(e)
        }
