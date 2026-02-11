from collections import deque
import time

class GestureBuffer:
    def __init__(self, window_seconds=1.0, fps=30):
        self.max_frames = int(window_seconds * fps)
        self.buffer = deque(maxlen=self.max_frames)

    def add(self, joints):
        timestamp = time.time()
        self.buffer.append((timestamp, joints))

    def is_full(self):
        return len(self.buffer) == self.max_frames

    def get_sequence(self):
        return [frame[1] for frame in self.buffer]

    def clear(self):
        self.buffer.clear()
