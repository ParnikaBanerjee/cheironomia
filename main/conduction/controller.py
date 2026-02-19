import numpy as np

class ConductionController:
    def __init__(self):
        self.prev_positions = []
        self.tempo = 90
        self.intensity = 0.5

    def update(self, segment):
        """
        segment: list of arm frames
        Returns: dict with tempo + intensity
        """

        if len(segment) < 5:
            return None

        y_positions = []
        velocities = []

        for i in range(1, len(segment)):
            prev = np.array(segment[i-1]["right_wrist"])
            curr = np.array(segment[i]["right_wrist"])

            y_positions.append(curr[1])
            velocities.append(np.linalg.norm(curr - prev))

        # --- Tempo estimation (oscillation frequency proxy) ---
        peaks = self.count_peaks(y_positions)
        duration = len(segment)

        freq = peaks / max(duration, 1)
        self.tempo = 60 + freq * 120   # scale

        # --- Intensity estimation ---
        avg_velocity = np.mean(velocities)
        self.intensity = min(avg_velocity * 10, 1.0)

        return {
            "tempo": round(self.tempo, 2),
            "intensity": round(self.intensity, 2)
        }

    def count_peaks(self, values):
        count = 0
        for i in range(1, len(values)-1):
            if values[i] > values[i-1] and values[i] > values[i+1]:
                count += 1
        return count
