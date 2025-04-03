import numpy as np
import matplotlib.pyplot as plt

# Example aggregated metrics for each configuration
metrics = {
    "seg2face": {"IoU": 1.00, "SSIM": 0.27, "Quality": 63.4},
    "seg2cat": {"IoU": 1.00, "SSIM": 0.44, "Quality": 72.2},
    "edge2car": {"IoU": 1.00, "SSIM": 0.74, "Quality": 86.7},
}

categories = list(next(iter(metrics.values())).keys())
num_vars = len(categories)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for cfg, vals in metrics.items():
    values = [vals[cat] * 100 if cat != "Quality" else vals[cat] for cat in categories]
    values += values[:1]
    ax.plot(angles, values, label=cfg, linewidth=2)
    ax.fill(angles, values, alpha=0.25)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), categories)
ax.set_ylim(0, 100)
ax.set_title("Accuracy Metrics by Configuration")
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.show()