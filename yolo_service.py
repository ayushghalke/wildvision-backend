"""
WildVision — YOLO Detection Service (ONNX Runtime)
Uses ONNX Runtime for lightweight inference (~50MB RAM instead of ~400MB).
No PyTorch dependency needed.
"""

import os
import numpy as np
from PIL import Image
import onnxruntime as ort

# Class names from the trained model
CLASS_NAMES = {
    0: 'Shih-Tzu',
    1: 'Rhodesian_ridgeback',
    2: 'beagle',
    3: 'English_foxhound',
    4: 'Border_terrier',
    5: 'Australian_terrier',
    6: 'golden_retriever',
    7: 'Old_English_sheepdog',
    8: 'Samoyed',
    9: 'dingo'
}

# Global session (lazy-loaded)
_session = None


def _get_session():
    """Lazy-load the ONNX model on first request."""
    global _session
    if _session is None:
        local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.onnx")
        parent_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "best.onnx")
        model_path = local_path if os.path.exists(local_path) else parent_path

        # Use CPU with minimal threads
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        _session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
    return _session


def _preprocess(image_path: str) -> np.ndarray:
    """Load and preprocess image for YOLO classification (224x224)."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr = np.expand_dims(arr, axis=0)  # Add batch dimension: (1, 3, 224, 224)
    return arr


def detect_animal(image_path: str) -> dict:
    """
    Run ONNX inference on the given image.
    Returns a dict with 'name', 'confidence', and 'all_detections'.
    """
    try:
        session = _get_session()
        input_data = _preprocess(image_path)

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_data})

        # outputs[0] shape: (1, num_classes) — raw logits or probabilities
        scores = outputs[0][0]

        # Apply softmax if needed (convert logits to probabilities)
        exp_scores = np.exp(scores - np.max(scores))
        probabilities = exp_scores / exp_scores.sum()

        # Build detections list
        detections = []
        for idx, prob in enumerate(probabilities):
            if prob > 0.01:  # Only include classes with >1% probability
                detections.append({
                    "name": CLASS_NAMES.get(idx, f"class_{idx}"),
                    "confidence": round(float(prob) * 100, 1)
                })

        # Sort by confidence, highest first
        detections.sort(key=lambda x: x["confidence"], reverse=True)

        if detections:
            best = detections[0]
            return {
                "name": best["name"],
                "confidence": best["confidence"],
                "all_detections": detections[:5]  # Top 5
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
