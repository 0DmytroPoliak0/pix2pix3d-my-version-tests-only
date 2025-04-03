import numpy as np
import matplotlib.pyplot as plt

# Example aggregated metrics for each configuration
metrics = {
    "seg2face": {"Avg Quality": 61.6, "Consistency": 96.1, "Performance": 100.0, "Overall": 85.2},
    "seg2cat": {"Avg Quality": 66.4, "Consistency": 94.6, "Performance": 100.0, "Overall": 86.3},
    "edge2car": {"Avg Quality": 85.6, "Consistency": 98.8, "Performance": 96.0, "Overall": 93.3},
}

categories = list(next(iter(metrics.values())).keys())
num_vars = len(categories)

# Function to create angles for the radar chart.
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # complete the loop

# Initialize radar chart.
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for cfg, vals in metrics.items():
    values = [vals[cat] for cat in categories]
    values += values[:1]  # complete the loop
    ax.plot(angles, values, label=cfg, linewidth=2)
    ax.fill(angles, values, alpha=0.25)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), categories)
ax.set_ylim(0, 100)
ax.set_title("Aggregated Musa Metrics by Configuration", y=1.08)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.show()