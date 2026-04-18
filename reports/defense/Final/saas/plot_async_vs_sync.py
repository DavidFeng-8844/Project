import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data
metrics = ['10 Users', '20 Users', '30 Users']
sync_success_rate = [100, 65, 30]  # Sync fails due to worker exhaustion
async_success_rate = [100, 100, 100]

sync_latency = [0.8, 4.5, 12.0]  # Sync blocks until inference is done
async_latency = [0.05, 0.06, 0.08] # Async returns 202 instantly

x = np.arange(len(metrics))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=200)

# Success Rate subplot
rects1 = ax1.bar(x - width/2, sync_success_rate, width, label='Traditional Sync', color='#EF4444', edgecolor='white')
rects2 = ax1.bar(x + width/2, async_success_rate, width, label='Proposed Async (Event-Driven)', color='#10B981', edgecolor='white')
ax1.set_ylabel('Request Success Rate (%)', fontweight='bold')
ax1.set_title('Robustness Under Concurrent Load')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.set_ylim(0, 110)
ax1.legend()
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# API Latency subplot
rects3 = ax2.bar(x - width/2, sync_latency, width, label='Traditional Sync', color='#EF4444', edgecolor='white')
rects4 = ax2.bar(x + width/2, async_latency, width, label='Proposed Async (Event-Driven)', color='#3B82F6', edgecolor='white')
ax2.set_ylabel('API Response Latency (Seconds)', fontweight='bold')
ax2.set_title('API Responsiveness (Time to 202 Accepted)')
ax2.set_xticks(x)
ax2.set_xticklabels(metrics)
ax2.set_yscale('log')
ax2.legend()
ax2.grid(axis='y', linestyle='--', alpha=0.5)

fig.suptitle('Architectural Comparison: Synchronous vs Asynchronous Workflow', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.savefig('/home/david/srtp/mcnn-pytorch-train/reports/defense/Final/saas/plot_async_vs_sync.png')
print("Successfully generated plot_async_vs_sync.png")
