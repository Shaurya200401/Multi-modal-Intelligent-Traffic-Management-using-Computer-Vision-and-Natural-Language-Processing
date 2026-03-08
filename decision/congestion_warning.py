from collections import deque

class CongestionWarning:
    def __init__(self, window=10):
        self.state_history = deque(maxlen=window)

    def update(self, traffic_state):
        self.state_history.append(traffic_state)

        # Rule-based early warning
        if self.state_history.count("BUILD_UP") >= 5:
            return "WARNING: Congestion building up"

        if self.state_history.count("UNSTABLE") >= 4:
            return "WARNING: Stop-and-go traffic detected"

        if self.state_history.count("CONGESTED") >= 3:
            return "ALERT: Congestion present"

        return None
