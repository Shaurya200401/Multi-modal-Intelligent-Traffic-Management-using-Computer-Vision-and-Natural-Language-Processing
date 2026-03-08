import cv2
import numpy as np

class BackgroundSubtraction:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=True
        )

    def get_foreground_ratio(self, frame):
        fg_mask = self.bg_subtractor.apply(frame)

        # Remove shadows (gray = 127)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        foreground_pixels = np.count_nonzero(fg_mask)
        total_pixels = fg_mask.size

        foreground_ratio = foreground_pixels / total_pixels
        return foreground_ratio, fg_mask  # <-- already perfect
