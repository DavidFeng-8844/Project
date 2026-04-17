import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data from Table 2.2
labels = ['BBox AP@50', 'Mask AP@50', 'Best F1 Score']
baseline = [0.799, 0.799, 0.850]
regularised = [0.907, 0.837, 0.878]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(8, 5), dpi=200)

# Beautiful colors
color_base = '#9CA3AF' # Gray
color_reg = '#2563EB'  # Blue

rects1 = ax.bar(x - width/2, baseline, width, label='Baseline Model', color=color_base, edgecolor='white', linewidth=1.5)
rects2 = ax.bar(x + width/2, regularised, width, label='Regularised Model\n(Backbone Frozen + Weight Decay)', color=color_reg, edgecolor='white', linewidth=1.5)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Score / Precision')
ax.set_title('Ablation Study: Baseline vs Regularised Model Performance')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontweight='bold')
ax.set_ylim(0.7, 1.0) # Zoom in to show the difference clearly
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add values on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

ax.legend(loc='upper left')

fig.tight_layout()
plt.savefig('/home/david/srtp/mcnn-pytorch-train/reports/defense/Final/ablation_bar_chart.png')
print("Bar chart saved successfully.")
