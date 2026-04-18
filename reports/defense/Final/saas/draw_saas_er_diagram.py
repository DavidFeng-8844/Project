import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis('off')

# Boxes
configs = [
    (1, 2.5, "Tenant", ["id (PK)", "name", "api_quota"], '#FEE2E2', '#DC2626'),
    (4.5, 2.5, "User", ["id (PK)", "tenant_id (FK)", "role", "email"], '#E0E7FF', '#4F46E5'),
    (8, 2.5, "InferenceTask", ["id (PK)", "tenant_id (FK)", "user_id (FK)", "status", "result"], '#D1FAE5', '#059669')
]

for x, y, title, attrs, fc, ec in configs:
    # Table Header
    ax.add_patch(patches.Rectangle((x-1.2, y+0.4), 2.4, 0.6, facecolor=ec, edgecolor=ec, zorder=2))
    ax.text(x, y+0.7, title, color='white', fontweight='bold', fontsize=12, ha='center', va='center', zorder=3)
    
    # Attributes
    h = len(attrs)*0.4 + 0.2
    ax.add_patch(patches.Rectangle((x-1.2, y+0.4-h), 2.4, h, facecolor=fc, edgecolor=ec, zorder=2))
    for i, attr in enumerate(attrs):
        if "(PK)" in attr:
            ax.text(x-1.0, y+0.1 - i*0.4, attr, fontsize=10, fontweight='bold', color='#111827', zorder=3)
        elif "(FK)" in attr:
            ax.text(x-1.0, y+0.1 - i*0.4, attr, fontsize=10, color='#991B1B', fontstyle='italic', zorder=3)
        else:
            ax.text(x-1.0, y+0.1 - i*0.4, attr, fontsize=10, color='#374151', zorder=3)

# Relationships
# Tenant -> User (1:N)
ax.annotate('', xy=(3.3, 2.7), xytext=(2.2, 2.7), 
            arrowprops=dict(facecolor='#4B5563', edgecolor='#4B5563', width=1.5, headwidth=6))
ax.text(2.75, 2.9, "1 : N", ha='center', fontsize=9, fontweight='bold')

# Tenant -> InferenceTask (1:N)
ax.annotate('', xy=(6.5, 3.8), xytext=(2.2, 3.8), 
            arrowprops=dict(facecolor='#4B5563', edgecolor='#4B5563', width=1.5, headwidth=6))
ax.plot([2.2, 2.2, 7.0, 7.0], [2.8, 3.8, 3.8, 2.8], color='#4B5563', lw=1.5, zorder=1)
ax.text(4.5, 4.0, "1 : N (Tenant Isolation Barrier)", ha='center', fontsize=9, fontweight='bold')

# User -> InferenceTask (1:N)
ax.annotate('', xy=(6.8, 2.7), xytext=(5.7, 2.7), 
            arrowprops=dict(facecolor='#4B5563', edgecolor='#4B5563', width=1.5, headwidth=6))
ax.text(6.25, 2.9, "1 : N", ha='center', fontsize=9, fontweight='bold')

fig.tight_layout()
plt.savefig('/home/david/srtp/mcnn-pytorch-train/reports/defense/Final/saas/figures/saas_er_diagram.png')
print("Generated saas_er_diagram.png")
