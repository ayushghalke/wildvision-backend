"""
WildVision — YOLO Detection Service
Loads the pre-trained best.pt model and runs inference on uploaded images.
"""

from ultralytics import YOLO
import os

# Load the pre-trained model — check local dir first, then parent dir
_LOCAL_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")
_PARENT_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "best.pt")
MODEL_PATH = _LOCAL_MODEL if os.path.exists(_LOCAL_MODEL) else _PARENT_MODEL
model = YOLO(MODEL_PATH)


def detect_animal(image_path: str) -> dict:
    """
    Run YOLO inference on the given image.
    Returns a dict with 'name', 'confidence', and 'all_detections'.
    """
    try:
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
