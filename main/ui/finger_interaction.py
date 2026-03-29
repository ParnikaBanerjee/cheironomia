"""
finger_interaction.py
Detects finger-based gestures (pointing, pinching) from hand landmarks.
Works with MediaPipe hand landmarks to extract pointing direction and pinch strength.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class FingerState:
    """State of finger interaction at a given frame."""
    index_pos: Tuple[float, float]  # (x, y) in normalized coordinates [0, 1]
    is_pointing: bool  # Index finger extended beyond threshold
    is_pinching: bool  # Thumb and index close together
    pinch_strength: float  # 0.0 (far) to 1.0 (touching)
    pointing_confidence: float  # How confident in the pointing detection


class FingerInteraction:
    """
    Detects hand gestures from MediaPipe hand landmarks.
    Assumes left hand is used for UI interaction.
    """

    def __init__(self, pointing_threshold: float = 0.05, pinch_threshold: float = 0.05):
        """
        Args:
            pointing_threshold: Distance threshold for index finger extension (0-1 scale)
            pinch_threshold: Distance threshold for pinch detection (0-1 scale)
        """
        self.pointing_threshold = pointing_threshold
        self.pinch_threshold = pinch_threshold
        
        # Landmark indices (MediaPipe hand model)
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.MIDDLE_TIP = 12
        self.RING_TIP = 16
        self.PINKY_TIP = 20
        
        self.THUMB_IP = 3  # Thumb interphalangeal joint
        self.INDEX_PIP = 6  # Index PIP joint
        self.MIDDLE_PIP = 10

    def process_hand_landmarks(self, hand_landmarks) -> Optional[FingerState]:
        """
        Process MediaPipe hand landmarks and extract finger state.
        
        Args:
            hand_landmarks: MediaPipe NormalizedLandmarkList from hands detection
            
        Returns:
            FingerState or None if landmarks are invalid
        """
        if hand_landmarks is None or len(hand_landmarks) < 21:
            return None

        try:
            # Extract key points
            wrist = np.array([hand_landmarks[self.WRIST].x, hand_landmarks[self.WRIST].y])
            thumb_tip = np.array([hand_landmarks[self.THUMB_TIP].x, hand_landmarks[self.THUMB_TIP].y])
            index_tip = np.array([hand_landmarks[self.INDEX_TIP].x, hand_landmarks[self.INDEX_TIP].y])
            middle_tip = np.array([hand_landmarks[self.MIDDLE_TIP].x, hand_landmarks[self.MIDDLE_TIP].y])
            thumb_ip = np.array([hand_landmarks[self.THUMB_IP].x, hand_landmarks[self.THUMB_IP].y])
            index_pip = np.array([hand_landmarks[self.INDEX_PIP].x, hand_landmarks[self.INDEX_PIP].y])

            # Detect pointing: index finger extended beyond middle finger
            index_to_middle_dist = np.linalg.norm(index_tip - middle_tip)
            index_extension = np.linalg.norm(index_tip - index_pip)
            middle_extension = np.linalg.norm(middle_tip - np.array([hand_landmarks[self.MIDDLE_PIP].x, hand_landmarks[self.MIDDLE_PIP].y]))
            
            is_pointing = (index_extension > middle_extension + self.pointing_threshold) and (index_to_middle_dist > 0.05)
            pointing_confidence = min((index_extension - middle_extension) / 0.2, 1.0)  # Normalize to 0-1

            # Detect pinch: thumb and index fingers close together
            thumb_index_dist = np.linalg.norm(thumb_tip - index_tip)
            is_pinching = thumb_index_dist < self.pinch_threshold
            pinch_strength = 1.0 - min(thumb_index_dist / self.pinch_threshold, 1.0)

            return FingerState(
                index_pos=(float(index_tip[0]), float(index_tip[1])),
                is_pointing=is_pointing,
                is_pinching=is_pinching,
                pinch_strength=pinch_strength,
                pointing_confidence=float(pointing_confidence)
            )

        except Exception as e:
            print(f"Error processing hand landmarks: {e}")
            return None

    def get_finger_position(self, state: FingerState) -> Tuple[float, float]:
        """Get normalized finger position (0-1 for x and y)."""
        return state.index_pos

    def is_pointing(self, state: FingerState) -> bool:
        """Check if index finger is pointing."""
        return state.is_pointing

    def is_pinching(self, state: FingerState) -> bool:
        """Check if thumb and index are pinching."""
        return state.is_pinching

    def get_pinch_strength(self, state: FingerState) -> float:
        """Get pinch strength (0.0 = far, 1.0 = touching)."""
        return state.pinch_strength

    def get_pointing_confidence(self, state: FingerState) -> float:
        """Get confidence in pointing detection."""
        return state.pointing_confidence
