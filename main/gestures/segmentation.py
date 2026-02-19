# gestures/segmentation.py

import numpy as np

class GestureSegmenter:
    def __init__(self,
                 start_threshold=0.01,
                 end_threshold=0.005,
                 min_frames=6):
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.min_frames = min_frames

        self.active = False
        self.current_segment = []
        self.prev_frame = None

    def compute_motion(self, frame):
        """
        Uses right wrist velocity as motion energy
        """
        if self.prev_frame is None:
            self.prev_frame = frame
            return 0.0

        prev = np.array(self.prev_frame["right_wrist"])
        curr = np.array(frame["right_wrist"])

        velocity = np.linalg.norm(curr - prev)

        self.prev_frame = frame
        return velocity

    def update(self, frame):
        """
        frame: dict of arm joints
        Returns:
            segment (list of frames) OR None
        """

        motion = self.compute_motion(frame)

        # Start condition
        if not self.active and motion > self.start_threshold:
            self.active = True
            self.current_segment = []

        # Collect frames if active
        if self.active:
            self.current_segment.append(frame)

        # End condition
        if self.active and motion < self.end_threshold:
            if len(self.current_segment) >= self.min_frames:
                segment = self.current_segment.copy()
                self.current_segment = []
                self.active = False
                return segment
            else:
                self.current_segment = []
                self.active = False

        return None
