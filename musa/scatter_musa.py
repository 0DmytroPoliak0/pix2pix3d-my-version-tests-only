import matplotlib.pyplot as plt
import numpy as np

# Example data (replace with your actual measured values)
data = {
    "seg2face": {
        "times": [8.50, 7.65, 7.38, 7.55, 7.32],
        "quality_scores": [63.3, 61.6, 63.8, 57.0, 62.4]
    },
    "seg2cat": {
        "times": [7.45, 7.51, 7.50, 7.36, 7.46],
        "quality_scores": [67.9, 71.3, 67.2, 65.1, 60.4]
    },
    "edge2car": {
        "times": [22.70, 22.74, 22.67, 22.73, 23.79],
        "quality_scores": [87.1, 85.1, 86.0, 85.6, 84.1]
    }
}

plt.figure(figsize=(10, 6))
for cfg, values in data.items():
    times = values["times"]
    quality = values["quality_scores"]
    plt.scatter(times, quality, s=100, label=cfg)

plt.xlabel("Execution Time (seconds)")
plt.ylabel("Quality Score (0-100)")
plt.title("Quality Score vs. Execution Time by Configuration")
plt.legend()
plt.grid(True)
plt.show()