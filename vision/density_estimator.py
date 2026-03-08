from vision.background_subtraction import BackgroundSubtraction
from vision.optical_flow import OpticalFlowEstimator
from vision.static_occupancy import StaticOccupancyEstimator


class TrafficDensityEstimator:
    def __init__(self):
        self.bg = BackgroundSubtraction()
        self.flow = OpticalFlowEstimator()
        self.static_occ = StaticOccupancyEstimator()

    def estimate_density(self, frame):
        # Moving vehicles
        fg_ratio, fg_mask = self.bg.get_foreground_ratio(frame)

        # Motion
        flow_mag = self.flow.compute_flow_magnitude(frame)

        # Static occupancy (stopped vehicles)
        static_occ, edge_mask = self.static_occ.estimate(frame)

        # ---------- Weighted density score ----------
        density_score = (
            (0.4 * fg_ratio) +
            (0.3 * flow_mag) +
            (0.3 * static_occ)
        )

        # ---------- Density level ----------
        if density_score < 0.12:
            level = "LOW"
        elif density_score < 0.30:
            level = "MEDIUM"
        else:
            level = "HIGH"

        return density_score, level, fg_mask, edge_mask
