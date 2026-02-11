# ui/visual_debug.py
import cv2

def draw(frame, data):
    h, w, _ = frame.shape

    # -------- DRAW HANDS --------
    for hand in data["hands"]:
        for lm in hand:
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    # -------- DRAW ARMS --------
    if data["arms"]:
        for joint in data["arms"].values():
            cx = int(joint[0] * w)
            cy = int(joint[1] * h)
            cv2.circle(frame, (cx, cy), 6, (255, 0, 0), -1)
