import cv2
import numpy as np

class OpticalFlowEstimator:
    def __init__(self):
        self.prev_gray = None

    def compute_flow_magnitude(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_magnitude = np.mean(magnitude)

        self.prev_gray = gray
        self.last_flow = flow  # <-- store for visualization
        return avg_magnitude

    def draw_flow(self, frame, step=16):
        if not hasattr(self, "last_flow"):
            return frame

        h, w = frame.shape[:2]
        vis = frame.copy()

        for y in range(0, h, step):
            for x in range(0, w, step):
                fx, fy = self.last_flow[y, x]
                cv2.arrowedLine(
                    vis,
                    (x, y),
                    (int(x + fx), int(y + fy)),
                    (0, 255, 0),
                    1,
                    tipLength=0.3
                )

        return vis
