"""
instrument_selector.py
Renders the instrument selection UI with buttons for Strings, Brass, Drums, Trumpet.
Handles button rendering, hover states, and coordinate mapping.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from enum import Enum


class ButtonState(Enum):
    """Visual state of a button."""
    NORMAL = 0
    HOVERING = 1
    SELECTED = 2


@dataclass
class Button:
    """Represents an instrument selection button."""
    name: str
    center: Tuple[int, int]  # (x, y) in pixel coordinates
    size: Tuple[int, int]  # (width, height) in pixels
    state: ButtonState = ButtonState.NORMAL
    
    def contains_point(self, point: Tuple[int, int], tolerance: float = 50) -> bool:
        """Check if point is near/on the button (with tolerance for finger size)."""
        x, y = point
        cx, cy = self.center
        w, h = self.size
        
        # Check if point is within button bounds + tolerance
        dist_x = abs(x - cx)
        dist_y = abs(y - cy)
        
        return dist_x <= (w / 2 + tolerance) and dist_y <= (h / 2 + tolerance)


class InstrumentSelector:
    """
    Manages and renders instrument selection buttons.
    Arranges 4 buttons in a 2x2 grid layout.
    """

    def __init__(self, screen_width: int = 640, screen_height: int = 480):
        """
        Args:
            screen_width: Width of the display window
            screen_height: Height of the display window
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Button dimensions
        self.button_width = 120
        self.button_height = 80
        self.button_spacing_x = 180
        self.button_spacing_y = 140
        
        # Calculate button positions (2x2 grid, right side of screen)
        margin_left = self.screen_width - 400
        margin_top = 80
        
        self.buttons: Dict[str, Button] = {
            "Strings": Button(
                name="Strings",
                center=(margin_left, margin_top),
                size=(self.button_width, self.button_height),
                state=ButtonState.NORMAL
            ),
            "Brass": Button(
                name="Brass",
                center=(margin_left + self.button_spacing_x, margin_top),
                size=(self.button_width, self.button_height),
                state=ButtonState.NORMAL
            ),
            "Drums": Button(
                name="Drums",
                center=(margin_left, margin_top + self.button_spacing_y),
                size=(self.button_width, self.button_height),
                state=ButtonState.NORMAL
            ),
            "Trumpet": Button(
                name="Trumpet",
                center=(margin_left + self.button_spacing_x, margin_top + self.button_spacing_y),
                size=(self.button_width, self.button_height),
                state=ButtonState.NORMAL
            ),
        }
        
        # Colors (BGR format for OpenCV)
        self.color_normal = (100, 100, 100)      # Gray
        self.color_hovering = (0, 255, 255)      # Yellow/Cyan
        self.color_selected = (0, 255, 0)        # Green
        self.color_text = (255, 255, 255)        # White
        self.color_border = (200, 200, 200)      # Light gray

    def set_window_size(self, width: int, height: int):
        """Update window dimensions and recalculate button positions."""
        self.screen_width = width
        self.screen_height = height
        self._recalculate_positions()

    def _recalculate_positions(self):
        """Recalculate button positions based on current window size."""
        margin_left = self.screen_width - 400
        margin_top = 80
        
        positions = {
            "Strings": (margin_left, margin_top),
            "Brass": (margin_left + self.button_spacing_x, margin_top),
            "Drums": (margin_left, margin_top + self.button_spacing_y),
            "Trumpet": (margin_left + self.button_spacing_x, margin_top + self.button_spacing_y),
        }
        
        for name, pos in positions.items():
            self.buttons[name].center = pos

    def update_button_state(self, name: str, state: ButtonState):
        """Update the visual state of a button."""
        if name in self.buttons:
            self.buttons[name].state = state

    def get_button_at_position(self, x: float, y: float) -> Optional[str]:
        """
        Find which button (if any) is at the given normalized position.
        
        Args:
            x: Normalized x coordinate (0-1)
            y: Normalized y coordinate (0-1)
            
        Returns:
            Button name or None if no button at position
        """
        # Convert normalized coordinates to pixel coordinates
        pixel_x = int(x * self.screen_width)
        pixel_y = int(y * self.screen_height)
        
        for name, button in self.buttons.items():
            if button.contains_point((pixel_x, pixel_y)):
                return name
        
        return None

    def get_all_button_names(self) -> List[str]:
        """Get list of all available instrument names."""
        return list(self.buttons.keys())

    def draw(self, frame: np.ndarray, finger_pos: Optional[Tuple[float, float]] = None, 
             is_pinching: bool = False, pinch_strength: float = 0.0) -> np.ndarray:
        """
        Draw the instrument selector UI on the frame.
        
        Args:
            frame: The video frame (BGR format)
            finger_pos: Normalized finger position (x, y) in [0, 1]
            is_pinching: Whether user is currently pinching
            pinch_strength: Pinch strength (0-1)
            
        Returns:
            Frame with UI rendered
        """
        frame_copy = frame.copy()
        
        # Draw each button
        for button in self.buttons.values():
            # Determine button color based on state
            color = self.color_normal
            if button.state == ButtonState.HOVERING:
                color = self.color_hovering
            elif button.state == ButtonState.SELECTED:
                color = self.color_selected
            
            # Draw button rectangle
            x, y = button.center
            w, h = button.size
            x1, y1 = x - w // 2, y - h // 2
            x2, y2 = x + w // 2, y + h // 2
            
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), self.color_border, 2)
            
            # Draw button text
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(button.name, font, 0.6, 1)[0]
            text_x = x - text_size[0] // 2
            text_y = y + text_size[1] // 2
            cv2.putText(frame_copy, button.name, (text_x, text_y), font, 
                       0.6, self.color_text, 1, cv2.LINE_AA)
        
        # Draw finger cursor if finger position provided
        if finger_pos is not None:
            finger_x = int(finger_pos[0] * self.screen_width)
            finger_y = int(finger_pos[1] * self.screen_height)
            
            # Draw finger position circle
            cursor_color = (0, 165, 255)  # Orange
            cv2.circle(frame_copy, (finger_x, finger_y), 15, cursor_color, 2)
            cv2.circle(frame_copy, (finger_x, finger_y), 5, cursor_color, -1)
        
        # Draw pinch indicator if pinching
        if is_pinching:
            pinch_text = f"Pinch: {pinch_strength:.0%}"
            cv2.putText(frame_copy, pinch_text, (self.screen_width - 200, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        return frame_copy

    def reset_button_states(self):
        """Reset all buttons to normal state."""
        for button in self.buttons.values():
            button.state = ButtonState.NORMAL
