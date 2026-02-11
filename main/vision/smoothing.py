class ExponentialSmoother:
    def __init__(self, alpha=0.7):
        self.alpha = alpha
        self.prev = {}

    def smooth(self, key, value):
        if key not in self.prev:
            self.prev[key] = value
            return value

        smoothed = tuple(
            self.alpha * value[i] + (1 - self.alpha) * self.prev[key][i]
            for i in range(len(value))
        )

        self.prev[key] = smoothed
        return smoothed
