# main.py
import cv2
from vision.camera import Camera
from vision.hand_pose import HandPoseTracker
from ui.visual_debug import draw

from gestures.normalize import normalize_sequence
from gestures.record import save_sequence
from conduction.controller import ConductionController
from gestures.segmentation import GestureSegmenter


def main():
    camera = Camera()
    tracker = HandPoseTracker()

    recording_label = None
    controller = ConductionController()
    segmenter = GestureSegmenter()

    while True:
        frame = camera.read()
        if frame is None:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        data = tracker.process(rgb)

        # -------- Segmentation --------
        segment = None
        if data["arms"]:
            segment = segmenter.update(data["arms"])

        if segment:

            # -------- Save for Training (Segment-based) --------
            if recording_label:
                norm_seq = normalize_sequence(segment)
                save_sequence(norm_seq, recording_label)
                print(f"[RECORDED] {recording_label}")
                recording_label = None  # reset after one save

            # -------- Conduction Controller --------
            result = controller.update(segment)
            if result:
                print("Tempo:", result["tempo"],
                      "Intensity:", result["intensity"])

        # -------- Visualization --------
        draw(frame, data)
        cv2.imshow("Virtual Orchestra – Gesture Capture", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('1'):
            recording_label = "start"
            print("Recording next segment as: start")

        elif key == ord('2'):
            recording_label = "stop"
            print("Recording next segment as: stop")

        elif key == ord('3'):
            recording_label = "accent"
            print("Recording next segment as: accent")

        elif key == ord('4'):
            recording_label = "cut"
            print("Recording next segment as: cut")

        elif key == ord('5'):
            recording_label = "pattern_change"
            print("Recording next segment as: pattern_change")

        elif key == ord('0'):
            recording_label = None
            print("Recording cancelled")

        elif key == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
