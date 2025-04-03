import matplotlib.pyplot as plt
import numpy as np

# Our test results (from your output)
results = {
    "seg2face": {
        "Avg Quality": 61.6,
        "Consistency": 96.1,
        "Performance": 100.0,
        "Overall Musa Score": 85.2
    },
    "seg2cat": {
        "Avg Quality": 66.4,
        "Consistency": 94.6,
        "Performance": 100.0,
        "Overall Musa Score": 86.3
    },
    "edge2car": {
        "Avg Quality": 85.6,
        "Consistency": 98.8,
        "Performance": 96.0,
        "Overall Musa Score": 93.3
    },
    "Weighted Overall": 87.2
}

# Plot Overall Musa Scores for each category
categories = ["seg2face", "seg2cat", "edge2car"]
overall_scores = [results[cat]["Overall Musa Score"] for cat in categories]

plt.figure(figsize=(8, 5))
bars = plt.bar(categories, overall_scores, color=['skyblue', 'lightgreen', 'salmon'])
plt.ylim(0, 100)
plt.ylabel("Overall Musa Score (0-100)")
plt.title("Musa Overall Scores by Category")

# Add data labels on top of each bar.
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}", ha='center', va='bottom')

# Plot weighted overall score as a horizontal line.
plt.axhline(y=results["Weighted Overall"], color='gray', linestyle='--', label=f"Weighted Overall: {results['Weighted Overall']:.1f}/100")
plt.legend()

plt.show()