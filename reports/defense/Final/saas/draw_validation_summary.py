import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Data
labels = ['Tenant Isolation (Security)', 'Task Lifecycle (State Machine)', 'Concurrency (Thread Safety)', 'Registry (Memory Safe)']
tests_run = [170, 145, 200, 32]
pass_rates = [100.0, 100.0, 100.0, 100.0]

fig, ax = plt.subplots(figsize=(9, 4), dpi=300)

y_pos = np.arange(len(labels))
bars = ax.barh(y_pos, tests_run, color=['#EF4444', '#3B82F6', '#10B981', '#8B5CF6'], edgecolor='white', height=0.6)

# Add text labels right of bars
for bar, pr in zip(bars, pass_rates):
    width = bar.get_width()
    ax.text(width + 5, bar.get_y() + bar.get_height()/2.,
            f"{int(width)} tests ({pr}% PASS)",
            ha='left', va='center', fontweight='bold', color='#1F2937')

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontweight='bold')
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Number of Automated Validation Sequences', fontweight='bold')
ax.set_title('SaaS Platform Systematic Validation Coverage', fontweight='bold', pad=15)
ax.set_xlim(0, max(tests_run) * 1.3)
ax.grid(axis='x', linestyle='--', alpha=0.5)

fig.tight_layout()
plt.savefig('/home/david/srtp/mcnn-pytorch-train/reports/defense/Final/saas/figures/saas_validation_coverage.png')
print("Generated saas_validation_coverage.png")
