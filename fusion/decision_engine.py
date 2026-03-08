class DecisionEngine:
    def __init__(self):
        pass

    def decide(self, density_level, density_score, incident, severity):
        final_score = density_score * 5 + severity

        if final_score < 2:
            action = "NORMAL_SIGNAL"
        elif final_score < 4:
            action = "EXTEND_GREEN"
        else:
            action = "EMERGENCY_OVERRIDE"

        return {
            "final_score": round(final_score, 2),
            "decision": action
        }
