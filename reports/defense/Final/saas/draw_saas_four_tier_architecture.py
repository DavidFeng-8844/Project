import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')

# Colors
clrs = ['#E0F2FE', '#FEF3C7', '#D1FAE5', '#FCE7F3']
edge_clrs = ['#0284C7', '#D97706', '#059669', '#DB2777']
labels = [
    ("Presentation Layer", "Frontend UI\n(JavaScript, HTML5, Bootstrap 5)"),
    ("Application Layer", "API Gateway & Session Control\n(Flask, JWT, Routes)"),
    ("Service Layer", "Task Orchestration\n(ThreadPoolExecutor, OpenCV, YOLO, Mask R-CNN)"),
    ("Data Layer", "Persistence & Tenant Limits\n(SQLAlchemy, SQLite/PostgreSQL)")
]

for i in range(4):
    y = 6 - i*1.8
    rect = patches.FancyBboxPatch((1, y), 8, 1.2, boxstyle="round,pad=0.1", 
                                  linewidth=2, edgecolor=edge_clrs[i], facecolor=clrs[i])
    ax.add_patch(rect)
    ax.text(1.5, y+0.85, labels[i][0], fontsize=14, fontweight='bold', color=edge_clrs[i])
    ax.text(1.5, y+0.4, labels[i][1], fontsize=11, color='#374151')

# Draw arrows between layers
for i in range(3):
    y1 = 6 - i*1.8
    y2 = y1 - 0.6
    ax.annotate('', xy=(5, y2), xytext=(5, y1), 
                arrowprops=dict(facecolor='#6B7280', edgecolor='#6B7280', width=2, headwidth=8))

fig.tight_layout()
plt.savefig('/home/david/srtp/mcnn-pytorch-train/reports/defense/Final/saas/figures/saas_four_tier_architecture.png')
print("Generated saas_four_tier_architecture.png")
