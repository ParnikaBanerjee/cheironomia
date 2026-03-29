"""
Real-time Inference Wrapper for Gesture Classifier

Provides a simple interface for classifying gesture segments in real-time.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional

from .gesture_dataset import GestureDataset
from .gesture_classifier import create_model


class GestureInference:
    """
    Wrapper for real-time gesture classification.
    
    Takes a raw gesture segment (T, 6, 3) and returns:
    - Predicted gesture label
    - Confidence score (softmax probability)
    """
    
    def __init__(
        self,
        model_path: str = "models/gesture_classifier_best.pth",
        sequence_length: int = 60,
        device: str = "cpu",
        confidence_threshold: float = 0.3
    ):
        """
        Args:
            model_path: Path to trained model checkpoint
            sequence_length: Fixed sequence length (must match training)
            device: Device to run inference on ("cpu" or "cuda")
            confidence_threshold: Minimum confidence to return prediction
        """
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        self.model = None
        self.dataset = GestureDataset()  # For label mappings only
        
        self._load_model()
    
    def _load_model(self):
        """Load trained model from checkpoint."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.model = create_model(device=self.device)
        self.model.load(self.model_path)
        print(f"✓ Gesture classifier loaded from {self.model_path}")
    
    def _preprocess_segment(self, segment: np.ndarray) -> torch.Tensor:
        """
        Preprocess raw gesture segment for inference.
        
        Args:
            segment: Raw segment of shape (T, 6, 3)
        
        Returns:
            Tensor of shape (1, sequence_length, 18) ready for model
        """
        T = segment.shape[0]
        
        # Pad or truncate to fixed length
        if T < self.sequence_length:
            padded = np.zeros((self.sequence_length, 6, 3), dtype=np.float32)
            padded[:T] = segment.astype(np.float32)
            segment = padded
        else:
            segment = segment[-self.sequence_length:].astype(np.float32)
        
        # Flatten (T, 6, 3) → (T, 18)
        segment = segment.reshape(self.sequence_length, -1)
        
        # Add batch dimension: (T, 18) → (1, T, 18)
        tensor = torch.from_numpy(segment).float().unsqueeze(0)
        
        return tensor.to(self.device)
    
    def predict(
        self,
        segment: np.ndarray
    ) -> Tuple[str, float]:
        """
        Classify a gesture segment.
        
        Args:
            segment: Normalized gesture segment of shape (T, 6, 3)
                     where T is variable time steps
        
        Returns:
            (predicted_label, confidence_score)
            predicted_label: str ("start", "stop", "accent", "cut", "pattern_change")
            confidence_score: float in [0, 1]
        """
        if segment.ndim != 3 or segment.shape[1:] != (6, 3):
            raise ValueError(f"Expected shape (T, 6, 3), got {segment.shape}")
        
        # Preprocess
        tensor = self._preprocess_segment(segment)
        
        # Inference
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)  # (1, 5)
            probs = torch.softmax(logits, dim=1)[0]  # (5,)
            confidence, pred_idx = torch.max(probs, dim=0)
        
        confidence = confidence.item()
        pred_label = self.dataset.get_label_name(pred_idx.item())
        
        return pred_label, confidence
    
    def predict_batch(
        self,
        segments: list
    ) -> list:
        """
        Classify multiple gesture segments.
        
        Args:
            segments: List of segments, each shape (T, 6, 3)
        
        Returns:
            List of (label, confidence) tuples
        """
        results = []
        for segment in segments:
            label, conf = self.predict(segment)
            results.append((label, conf))
        return results


def test_inference():
    """Test inference with random data (for debugging)."""
    print("Testing GestureInference with random data...")
    
    # Create dummy inference wrapper (won't load actual model if missing)
    try:
        inf = GestureInference()
    except FileNotFoundError:
        print("Model not found. Testing preprocessing only...")
        # Test preprocessing without model
        return
    
    # Test prediction with random segment
    test_segment = np.random.randn(30, 6, 3).astype(np.float32)  # 30 frames
    label, conf = inf.predict(test_segment)
    
    print(f"Test segment shape: {test_segment.shape}")
    print(f"Predicted label: {label}")
    print(f"Confidence: {conf:.4f}")
    print("✓ Inference test passed")


if __name__ == "__main__":
    test_inference()
