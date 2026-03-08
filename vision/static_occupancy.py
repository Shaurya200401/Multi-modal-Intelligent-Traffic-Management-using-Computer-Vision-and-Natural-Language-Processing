import cv2
import numpy as np

class StaticOccupancyEstimator:
    def __init__(self):
        pass

    def estimate(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Edge detection picks up vehicle boundaries
        edges = cv2.Canny(gray, 80, 160)

        # Occupancy = edge density
        occupancy_ratio = np.count_nonzero(edges) / edges.size

        return occupancy_ratio, edges
