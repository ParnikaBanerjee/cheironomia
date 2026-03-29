"""
selection_manager.py
Manages instrument selection state with debouncing and pinch confirmation.
Handles hover detection, selection confirmation, and cooldown logic.
"""

import time
from typing import Optional, Callable, List, Dict
from dataclasses import dataclass
from enum import Enum


class SelectionEvent(Enum):
    """Types of selection events."""
    HOVER = "hover"
    CONFIRMED = "confirmed"
    DESELECTED = "deselected"


@dataclass
class SelectionRecord:
    """Record of a selection event."""
    instrument: str
    timestamp: float
    event_type: SelectionEvent
    pinch_strength: float


class SelectionManager:
    """
    Manages instrument selection state with debouncing and confirmation.
    Uses pinch-and-hold for confirmation to prevent accidental selections.
    """

    def __init__(self, pinch_hold_duration: float = 0.3, debounce_duration: float = 0.5):
        """
        Args:
            pinch_hold_duration: Duration to hold pinch before confirming selection (seconds)
            debounce_duration: Minimum time between selections (seconds)
        """
        self.pinch_hold_duration = pinch_hold_duration
        self.debounce_duration = debounce_duration
        
        # State tracking
        self.current_instrument: Optional[str] = None
        self.hovered_instrument: Optional[str] = None
        self.pinch_start_time: Optional[float] = None
        self.last_selection_time: float = 0.0
        
        # Event callbacks
        self._selection_callbacks: List[Callable[[SelectionRecord], None]] = []
        
        # Selection history for debugging
        self.selection_history: List[SelectionRecord] = []
        self.max_history_size = 100

    def update(self, hovered_instrument: Optional[str], is_pinching: bool, 
               pinch_strength: float = 0.0):
        """
        Update selection state based on current hand state.
        
        Args:
            hovered_instrument: Name of instrument under finger (or None)
            is_pinching: Whether user is currently pinching
            pinch_strength: Strength of pinch (0-1)
        """
        current_time = time.time()
        
        # Handle hover state
        if hovered_instrument != self.hovered_instrument:
            self.hovered_instrument = hovered_instrument
            if hovered_instrument:
                self._emit_event(hovered_instrument, SelectionEvent.HOVER, pinch_strength)
        
        # Handle pinch confirmation
        if is_pinching and self.hovered_instrument:
            # Start pinch hold timing if not already timing
            if self.pinch_start_time is None:
                self.pinch_start_time = current_time
            
            # Check if pinch held long enough for confirmation
            pinch_duration = current_time - self.pinch_start_time
            if pinch_duration >= self.pinch_hold_duration:
                # Check debounce
                time_since_last = current_time - self.last_selection_time
                if time_since_last >= self.debounce_duration:
                    self._confirm_selection(self.hovered_instrument, current_time, pinch_strength)
        else:
            # Pinch released
            self.pinch_start_time = None
    
    def _confirm_selection(self, instrument: str, timestamp: float, pinch_strength: float):
        """Confirm a selection and update state."""
        self.current_instrument = instrument
        self.last_selection_time = timestamp
        self.pinch_start_time = None  # Reset pinch timing
        
        self._emit_event(instrument, SelectionEvent.CONFIRMED, pinch_strength)
    
    def _emit_event(self, instrument: str, event_type: SelectionEvent, 
                    pinch_strength: float = 0.0):
        """Emit a selection event to all registered callbacks."""
        record = SelectionRecord(
            instrument=instrument,
            timestamp=time.time(),
            event_type=event_type,
            pinch_strength=pinch_strength
        )
        
        # Add to history
        self.selection_history.append(record)
        if len(self.selection_history) > self.max_history_size:
            self.selection_history.pop(0)
        
        # Call all registered callbacks
        for callback in self._selection_callbacks:
            try:
                callback(record)
            except Exception as e:
                print(f"Error in selection callback: {e}")
    
    def register_callback(self, callback: Callable[[SelectionRecord], None]):
        """
        Register a callback function to be called on selection events.
        
        Args:
            callback: Function that receives SelectionRecord
        """
        self._selection_callbacks.append(callback)
    
    def get_current_instrument(self) -> Optional[str]:
        """Get the currently selected instrument."""
        return self.current_instrument
    
    def get_hovered_instrument(self) -> Optional[str]:
        """Get the instrument currently being hovered over."""
        return self.hovered_instrument
    
    def get_pinch_progress(self) -> float:
        """
        Get progress of pinch hold (0.0 to 1.0).
        Returns 0.0 if not pinching, 1.0+ if pinch held long enough.
        """
        if self.pinch_start_time is None:
            return 0.0
        
        elapsed = time.time() - self.pinch_start_time
        progress = elapsed / self.pinch_hold_duration
        return min(progress, 1.0)  # Cap at 1.0
    
    def get_selection_history(self) -> List[SelectionRecord]:
        """Get the selection history for debugging."""
        return self.selection_history.copy()
    
    def clear_history(self):
        """Clear the selection history."""
        self.selection_history.clear()
    
    def reset(self):
        """Reset all state."""
        self.current_instrument = None
        self.hovered_instrument = None
        self.pinch_start_time = None
        self.last_selection_time = time.time()
