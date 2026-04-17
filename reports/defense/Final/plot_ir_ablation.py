import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

stages = ['No Suppression\n(Baseline)', 'Border Mask\nOnly', 'Border + OCR Masks\n(Proposed)']
fps = [18, 9, 4]

fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
colors = ['#EF4444', '#F59E0B', '#10B981']

bars = ax.bar(stages, fps, color=colors, edgecolor='white', linewidth=1.5, width=0.5)

# Add text above bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height}\nFalse Positives',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5), textcoords="offset points",
                ha='center', va='bottom', fontweight='bold')

ax.set_ylabel('Number of False Positives (per 50 images)', fontweight='bold')
ax.set_title('Ablation Study: Effectiveness of Suppression Masks')
ax.set_ylim([0, 24])
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Adding lines and arrows for percentage drop annotations
ax.annotate('-50% FP', xy=(0.5, 12), xytext=(0.5, 16), ha='center',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='black', lw=1.5), fontweight='bold')
ax.annotate('-78% FP Overall', xy=(1.5, 12), xytext=(1.5, 20), ha='center',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='black', lw=1.5), fontweight='bold')

fig.tight_layout()
plt.savefig('/home/david/srtp/mcnn-pytorch-train/reports/defense/Final/ir_fp_ablation.png')
print("Bar chart saved.")
