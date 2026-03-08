import matplotlib.pyplot as plt

STATE_COLOR_MAP = {
    "FREE_FLOW": "green",
    "STABLE_DENSE": "blue",
    "BUILD_UP": "orange",
    "CONGESTED": "red",
    "DISSIPATING": "purple",
    "WARMING_UP": "gray"
}

def plot_traffic_states(logger):
    plt.figure(figsize=(10, 5))

    # Plot density score
    plt.plot(logger.frames, logger.density_scores, label="Density Score", color="black")

    # Overlay traffic states
    for i, state in enumerate(logger.states):
        plt.scatter(
            logger.frames[i],
            logger.density_scores[i],
            color=STATE_COLOR_MAP.get(state, "black"),
            s=10
        )

    plt.xlabel("Frame Index (Time)")
    plt.ylabel("Traffic Density Score")
    plt.title("Traffic State Evolution Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
