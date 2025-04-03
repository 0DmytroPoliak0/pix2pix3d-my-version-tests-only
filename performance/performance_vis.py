import matplotlib.pyplot as plt

# Example data
test_names = ['Test 1', 'Test 2', 'Test 3']
execution_times = [9.2, 8.9, 9.1]  # seconds
memory_usages = [500, 550, 520]     # MB

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Test Cases')
ax1.set_ylabel('Execution Time (s)', color=color)
ax1.bar(test_names, execution_times, color=color, alpha=0.7)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Memory Usage (MB)', color=color)
ax2.plot(test_names, memory_usages, color=color, marker='o', linestyle='--')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Performance Metrics by Test Case')
plt.show()