import matplotlib.pyplot as plt
import numpy as np

# Updated data with new numbers
cpu_times = [16.81, 13.05, 9.26, 13.21, 9.92, 13.63, 6.93, 20.81, 10.14, 14.38]
cpu_speeds = [17.67, 18.39, 18.47, 17.94, 18.04, 17.98, 18.60, 17.88, 18.63, 18.15]

gpu_times = [3.33, 5.67, 3.07, 3.47, 9.45, 5.25, 4.53, 3.31, 4.91, 2.36]
gpu_speeds = [54.95, 55.58, 55.13, 54.98, 55.04, 55.64, 55.44, 54.68, 55.57, 54.67]

# Generate x-axis labels (run numbers)
runs = np.arange(1, len(cpu_times) + 1)

# Create a figure for the plots
plt.figure(figsize=(14, 7))

# Plot inference times
plt.subplot(1, 2, 1)
plt.plot(runs, cpu_times, linewidth=1.5, linestyle="-", color="#1f77b4", marker="o", label="CPU (FP32)")
plt.plot(runs, gpu_times, linewidth=1.5, linestyle="-", color="#2ca02c", marker="o", label="GPU (FP16)")
plt.title("Inference Time Comparison", fontsize=16, weight="bold")
plt.xlabel("Run Number", fontsize=14)
plt.ylabel("Inference Time (seconds)", fontsize=14)
plt.xticks(runs, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
plt.legend(fontsize=12, loc="upper right")

# Plot inference speed
plt.subplot(1, 2, 2)
plt.plot(runs, cpu_speeds, linewidth=1.5, linestyle="-", color="#1f77b4", marker="o", label="CPU (FP32)")
plt.plot(runs, gpu_speeds, linewidth=1.5, linestyle="-", color="#2ca02c", marker="o", label="GPU (FP16)")
plt.title("Inference Speed Comparison", fontsize=16, weight="bold")
plt.xlabel("Run Number", fontsize=14)
plt.ylabel("Speed (tokens/sec)", fontsize=14)
plt.xticks(runs, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
plt.legend(fontsize=12, loc="center right")

# Layout adjustments and save the plot as PNG
plt.tight_layout()
plt.savefig("inference_comparison_plot.png", dpi=300)
plt.show()