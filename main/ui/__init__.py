"""
UI package for Cheironomia Virtual Orchestra.
Provides visualization and finger-based instrument selection interface.
"""

from .finger_interaction import FingerInteraction, FingerState
from .instrument_selector import InstrumentSelector, ButtonState
from .selection_manager import SelectionManager, SelectionEvent, SelectionRecord
from .visual_debug import VisualDebugger, get_debugger

__all__ = [
    "FingerInteraction",
    "FingerState",
    "InstrumentSelector",
    "ButtonState",
    "SelectionManager",
    "SelectionEvent",
    "SelectionRecord",
    "VisualDebugger",
    "get_debugger",
]
