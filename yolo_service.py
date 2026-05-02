"""
WildVision — Dog Breed Detection Service (PyTorch MobileNetV2)
Uses a fine-tuned MobileNetV2 model (.pth) for 20-breed classification.
"""

import os
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

# 20 breed classes (alphabetical — matches ImageFolder ordering used during training)
CLASS_NAMES = {
    0: 'Beagle',
    1: 'Bloodhound',
    2: 'Border Collie',
    3: 'Border Terrier',
    4: 'Boston Bull',
    5: 'Boxer',
    6: 'Bull Mastiff',
    7: 'Chihuahua',
    8: 'Cocker Spaniel',
    9: 'Doberman',
    10: 'German Shepherd',
    11: 'Golden Retriever',
    12: 'Great Dane',
    13: 'Labrador Retriever',
    14: 'Pomeranian',
    15: 'Pug',
    16: 'Rottweiler',
    17: 'Saint Bernard',
    18: 'Shih Tzu',
    19: 'Siberian Husky',
}

# Standard ImageNet normalization (used during MobileNetV2 training)
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),                # Converts to [0, 1] and CHW
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Global model (lazy-loaded)
_model = None


def _get_model():
    """Lazy-load the MobileNetV2 model on first request."""
    global _model
    if _model is None:
        # Build MobileNetV2 architecture with 20 output classes
        _model = models.mobilenet_v2(weights=None)
        _model.classifier[1] = torch.nn.Linear(1280, 20)

        # Load trained weights
        local_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "dog_breed_model.pth"
        )
        state_dict = torch.load(local_path, map_location="cpu", weights_only=False)
        _model.load_state_dict(state_dict)

        # Set to evaluation mode (disables dropout, uses running BatchNorm stats)
        _model.eval()
    return _model


def detect_animal(image_path: str) -> dict:
    """
    Run MobileNetV2 inference on the given image.
    Returns a dict with 'name', 'confidence', and 'all_detections'.
    """
    try:
        model = _get_model()

        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        input_tensor = _transform(img).unsqueeze(0)  # Add batch dim: (1, 3, 224, 224)

        # Run inference (no gradient computation needed)
        with torch.no_grad():
            outputs = model(input_tensor)            # (1, 20) raw logits
            probabilities = F.softmax(outputs, dim=1)[0]  # (20,) probabilities

        # Build detections list
        detections = []
        for idx, prob in enumerate(probabilities):
            conf = round(float(prob) * 100, 1)
            if conf > 1.0:  # Only include classes with >1% probability
                detections.append({
                    "name": CLASS_NAMES.get(idx, f"class_{idx}"),
                    "confidence": conf,
                })

        # Sort by confidence, highest first
        detections.sort(key=lambda x: x["confidence"], reverse=True)

        if detections:
            best = detections[0]
            return {
                "name": best["name"],
                "confidence": best["confidence"],
                "all_detections": detections[:5],  # Top 5
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
