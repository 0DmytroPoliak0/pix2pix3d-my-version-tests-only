import matplotlib.pyplot as plt

# Example data: overall Musa scores per configuration.
configurations = ['seg2face', 'seg2cat', 'edge2car']
overall_scores = [85.2, 86.3, 64.6]  # from your Musa test results
performance_scores = [100.0, 100.0, 96.0]

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(configurations, overall_scores, color='skyblue')
for i, score in enumerate(overall_scores):
    ax.text(i, score + 1, f'{score:.1f}', ha='center')
ax.set_ylim(0, 100)
ax.set_xlabel("Configuration")
ax.set_ylabel("Overall Musa Score (0-100)")
ax.set_title("Overall Musa Scores by Configuration")
plt.show()