class TrafficLogger:
    def __init__(self):
        self.frames = []
        self.density_scores = []
        self.states = []

    def log(self, frame_id, density_score, traffic_state):
        self.frames.append(frame_id)
        self.density_scores.append(density_score)
        self.states.append(traffic_state)
