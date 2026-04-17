import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

thresholds = [40, 45, 50, 55, 60]
precision = [0.72, 0.81, 0.89, 0.94, 0.97]
recall = [0.95, 0.90, 0.85, 0.75, 0.60]
f1 = [0.82, 0.85, 0.87, 0.83, 0.74]

fig, ax1 = plt.subplots(figsize=(8, 5), dpi=200)

ax1.set_xlabel('Alarm Temperature Threshold (°C)', fontweight='bold')
ax1.set_ylabel('Score / Metric', fontweight='bold')

ax1.plot(thresholds, precision, marker='o', label='Precision', color='#2563EB', linewidth=2, markersize=8)
ax1.plot(thresholds, recall, marker='s', label='Recall', color='#10B981', linewidth=2, markersize=8)
ax1.plot(thresholds, f1, marker='^', linestyle='--', label='F1 Score', color='#EF4444', linewidth=3, markersize=10)

# Highlight Best F1
ax1.axvline(x=50, color='gray', linestyle=':', linewidth=1.5)
ax1.scatter([50], [0.87], color='#EF4444', s=150, zorder=5)
ax1.annotate('Best F1 (0.87)\n@ 50°C', xy=(50, 0.87), xytext=(51, 0.91),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
            fontweight='bold', color='#B91C1C')

ax1.set_ylim([0.5, 1.05])
ax1.set_xticks(thresholds)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='lower left', frameon=True)

plt.title('Precision, Recall, and F1 Score vs. Alarm Threshold')
fig.tight_layout()
plt.savefig('/home/david/srtp/mcnn-pytorch-train/reports/defense/Final/ir_threshold_metrics.png')
print("Line chart saved.")
