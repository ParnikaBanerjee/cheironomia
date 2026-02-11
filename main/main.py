# main.py
import cv2
from vision.camera import Camera
from vision.hand_pose import HandPoseTracker
from ui.visual_debug import draw

from gestures.buffer import GestureBuffer
from gestures.normalize import normalize_sequence
from gestures.record import save_sequence



def main():
    camera = Camera()
    tracker = HandPoseTracker()
    gesture_buffer = GestureBuffer(window_seconds=1.0, fps=30)
    recording_label = None

    while True:
        frame = camera.read()
        if frame is None:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        data = tracker.process(rgb)
        if data["arms"]:
            gesture_buffer.add(data["arms"])
        if gesture_buffer.is_full():
            sequence = gesture_buffer.get_sequence()
            norm_seq = normalize_sequence(sequence)

            if recording_label:
                save_sequence(norm_seq, recording_label)

            gesture_buffer.clear()


        # DEBUG: print arm joint coordinates
        if data["arms"]:
            arm_coords = {
                k: (round(v[0], 3), round(v[1], 3), round(v[2], 3))
                for k, v in data["arms"].items()
            }
            # print(arm_coords)

        draw(frame, data)

        cv2.imshow("Virtual Orchestra – Gesture Capture", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('1'):
            recording_label = "tempo_up"
            print("Recording: tempo_up")

        elif key == ord('2'):
            recording_label = "tempo_down"
            print("Recording: tempo_down")

        elif key == ord('3'):
            recording_label = "crescendo"
            print("Recording: crescendo")

        elif key == ord('0'):
            recording_label = None
            print("Recording stopped")

        elif key == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
