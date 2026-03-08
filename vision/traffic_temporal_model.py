from collections import deque
import numpy as np

class TrafficTemporalModel:
    def __init__(self, window_size=15, alpha=0.1):
        """
        window_size : number of recent frames to consider
        alpha       : learning rate for adaptive baseline
        """
        self.window_size = window_size
        self.density_window = deque(maxlen=window_size)

        # Adaptive (online learned) baseline
        self.baseline_mean = None
        self.baseline_std = None
        self.alpha = alpha

    def update_baseline(self, value):
        """Exponential moving average learning"""
        if self.baseline_mean is None:
            self.baseline_mean = value
            self.baseline_std = 0.0
        else:
            diff = value - self.baseline_mean
            self.baseline_mean += self.alpha * diff
            self.baseline_std = (1 - self.alpha) * self.baseline_std + self.alpha * abs(diff)

    def classify_state(self, density_score):
        self.density_window.append(density_score)
        self.update_baseline(density_score)

        # Not enough data yet
        if len(self.density_window) < 5:
            return "WARMING_UP"

        mean_density = np.mean(self.density_window)

        # Trend (slope)
        x = np.arange(len(self.density_window))
        slope = np.polyfit(x, list(self.density_window), 1)[0]

        # Adaptive thresholds
        high_thresh = self.baseline_mean + 2 * self.baseline_std
        low_thresh  = self.baseline_mean - 2 * self.baseline_std

        # ---- Traffic State Logic ----
        if mean_density > high_thresh and slope >= 0:
            return "CONGESTED"
        elif slope > 0.01:
            return "BUILD_UP"
        elif slope < -0.01:
            return "DISSIPATING"
        elif mean_density < low_thresh:
            return "FREE_FLOW"
        else:
            return "STABLE_DENSE"
