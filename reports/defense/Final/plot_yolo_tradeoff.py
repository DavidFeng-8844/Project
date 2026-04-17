import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Comparative data for YOLOv8 variants (Speed vs Accuracy trade-off on Oil Leak task)
models = ['YOLOv8n', 'YOLOv8s (Proposed)', 'YOLOv8m', 'YOLOv8l']
# parameters (Millions)
params = [3.2, 11.2, 25.9, 43.7]
# mAP50 (%)
map50 = [0.32, 0.39, 0.40, 0.38] # YOLO-l overfits due to small dataset
# Inference Speed (FPS on consumer GPU)
fps = [145, 120, 75, 45]

fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

colors = ['#9CA3AF', '#EF4444', '#60A5FA', '#3B82F6']
sizes = [p * 30 for p in params] # Bubble size proportional to parameter count

scatter = ax.scatter(fps, map50, s=sizes, c=colors, alpha=0.7, edgecolors='white', linewidth=2)

for i, model in enumerate(models):
    ax.annotate(model, (fps[i], map50[i]), xytext=(0, 15), 
                textcoords='offset points', ha='center', va='bottom', fontweight='bold')

ax.set_xlabel('Inference Speed (FPS) → Higher is better')
ax.set_ylabel('mAP@50 → Higher is better')
ax.set_title('Comparative Analysis: YOLOv8 Variants Trade-off')
ax.grid(True, linestyle='--', alpha=0.5)

# Add a text box explaining the bubble size
textstr = 'Bubble size = Parameter Count (Model Size)'
props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

fig.tight_layout()
plt.savefig('/home/david/srtp/mcnn-pytorch-train/reports/defense/Final/yolo_tradeoff_scatter.png')
print("Scatter plot saved successfully.")
