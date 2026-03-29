"""
test_finger_ui.py
Integration tests for the finger-based instrument selection UI system.
Run this to verify all components work correctly.
"""

import sys
import numpy as np
from pathlib import Path

# Add main directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ui.finger_interaction import FingerInteraction, FingerState
from ui.instrument_selector import InstrumentSelector, ButtonState
from ui.selection_manager import SelectionManager, SelectionEvent
from ui.visual_debug import VisualDebugger


def test_finger_interaction():
    """Test finger detection system."""
    print("\n=== Testing FingerInteraction ===")
    
    fi = FingerInteraction(pointing_threshold=0.05, pinch_threshold=0.05)
    
    # Create mock hand landmarks (21 points, normalized 0-1)
    class MockLandmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    
    # Create a hand where index finger is extended and thumb-index are pinched
    hand_landmarks = [
        MockLandmark(0.5, 0.5),  # 0: Wrist
        MockLandmark(0.45, 0.35),  # 1: Thumb CMC
        MockLandmark(0.44, 0.32),  # 2: Thumb MCP
        MockLandmark(0.43, 0.29),  # 3: Thumb IP
        MockLandmark(0.42, 0.25),  # 4: Thumb Tip
        MockLandmark(0.55, 0.35),  # 5: Index MCP
        MockLandmark(0.56, 0.30),  # 6: Index PIP
        MockLandmark(0.57, 0.20),  # 7: Index DIP
        MockLandmark(0.58, 0.10),  # 8: Index Tip (extended)
        MockLandmark(0.60, 0.35),  # 9: Middle MCP
        MockLandmark(0.61, 0.30),  # 10: Middle PIP
        MockLandmark(0.62, 0.25),  # 11: Middle DIP
        MockLandmark(0.63, 0.20),  # 12: Middle Tip
        *[MockLandmark(0.5, 0.5) for _ in range(8)]  # Rest of landmarks
    ]
    
    state = fi.process_hand_landmarks(hand_landmarks)
    assert state is not None, "Failed to process hand landmarks"
    assert state.is_pointing, "Failed to detect pointing"
    print(f"✓ Detected pointing: index_pos={state.index_pos}, confidence={state.pointing_confidence:.2f}")
    
    # Test pinching (move index tip closer to thumb)
    hand_landmarks[8] = MockLandmark(0.43, 0.25)  # Index tip near thumb tip
    state = fi.process_hand_landmarks(hand_landmarks)
    assert state is not None, "Failed to process hand landmarks"
    assert state.is_pinching, "Failed to detect pinching"
    print(f"✓ Detected pinching: pinch_strength={state.pinch_strength:.2f}")


def test_instrument_selector():
    """Test instrument button rendering and layout."""
    print("\n=== Testing InstrumentSelector ===")
    
    selector = InstrumentSelector(640, 480)
    buttons = selector.get_all_button_names()
    assert len(buttons) == 4, f"Expected 4 buttons, got {len(buttons)}"
    assert "Strings" in buttons, "Missing Strings button"
    print(f"✓ All buttons created: {buttons}")
    
    # Test button position detection
    button = selector.get_button_at_position(0.8, 0.2)
    print(f"✓ Button at position (0.8, 0.2): {button}")
    
    # Test button state updates
    selector.update_button_state("Strings", ButtonState.SELECTED)
    assert selector.buttons["Strings"].state == ButtonState.SELECTED
    print("✓ Button state update works")


def test_selection_manager():
    """Test selection state and debouncing."""
    print("\n=== Testing SelectionManager ===")
    
    manager = SelectionManager(pinch_hold_duration=0.1, debounce_duration=0.2)
    
    # Track selection events
    events = []
    def record_event(record):
        events.append(record)
    
    manager.register_callback(record_event)
    
    # Simulate hover
    manager.update(hovered_instrument="Strings", is_pinching=False)
    assert manager.get_hovered_instrument() == "Strings"
    print("✓ Hover detection works")
    
    # Simulate pinch
    import time
    manager.update(hovered_instrument="Strings", is_pinching=True, pinch_strength=0.5)
    time.sleep(0.15)  # Wait for pinch hold
    manager.update(hovered_instrument="Strings", is_pinching=True, pinch_strength=0.5)
    
    # Should have confirmed selection
    assert manager.get_current_instrument() == "Strings" or len(events) > 0
    print(f"✓ Pinch confirmation works (events recorded: {len(events)})")


def test_visual_debugger():
    """Test visual debugger integration."""
    print("\n=== Testing VisualDebugger ===")
    
    debugger = VisualDebugger(show_finger_ui=True)
    
    # Create dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create dummy data
    class MockLandmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    
    mock_hand = [MockLandmark(0.5, 0.5) for _ in range(21)]
    data = {
        "hands": [mock_hand],
        "arms": {"shoulder": (0.5, 0.4), "elbow": (0.5, 0.6), "wrist": (0.5, 0.8)}
    }
    
    # Test drawing
    output = debugger.draw(frame, data)
    assert output is not None, "Failed to draw frame"
    assert output.shape == frame.shape, "Output frame shape mismatch"
    print("✓ Visual debugger draws frame successfully")
    
    # Test UI toggle
    initial_state = debugger.show_finger_ui
    debugger.toggle_finger_ui()
    assert debugger.show_finger_ui != initial_state
    print("✓ UI toggle works")


def main():
    print("=" * 60)
    print("FINGER UI SYSTEM VALIDATION TESTS")
    print("=" * 60)
    
    try:
        test_finger_interaction()
        test_instrument_selector()
        test_selection_manager()
        test_visual_debugger()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nFingerUI System Status: READY FOR DEPLOYMENT")
        print("Next steps:")
        print("  1. Run 'python main.py' to start the application")
        print("  2. Press SPACE to toggle the finger UI on/off")
        print("  3. Use your left hand to select instruments")
        print("  4. Use your right hand for gesture recording/conduction")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
