"""
Generate Gantt chart for project schedule
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import numpy as np
from matplotlib.dates import DateFormatter, MonthLocator

# Start date: November 2025 (assuming week 1 starts on Nov 1, 2025)
start_date = datetime(2025, 11, 1)

# Define tasks with week numbers
tasks = [
    {
        'name': 'PoC Implementation (Module A: Bird Nest)',
        'start_week': 1,
        'end_week': 4,
        'status': 'completed',
        'color': '#2ecc71'  # Green for completed
    },
    {
        'name': 'Track 1: SaaS Platform Build (RBAC, Model Registry)',
        'start_week': 5,
        'end_week': 10,
        'status': 'planned',
        'color': '#3498db'  # Blue for planned
    },
    {
        'name': 'Track 2a: Module B (Oil Leak Detection)',
        'start_week': 11,
        'end_week': 15,
        'status': 'planned',
        'color': '#3498db'
    },
    {
        'name': 'Track 2b: Module C (Infrared High-Temperature)',
        'start_week': 16,
        'end_week': 18,
        'status': 'planned',
        'color': '#3498db'
    },
    {
        'name': 'Final Integration & Report',
        'start_week': 19,
        'end_week': 20,
        'status': 'planned',
        'color': '#3498db'
    }
]

# Convert week numbers to dates
def week_to_date(week_num):
    """Convert week number to date (assuming each week is 7 days)"""
    return start_date + timedelta(weeks=week_num - 1)

# Convert tasks to use dates
for task in tasks:
    task['start_date'] = week_to_date(task['start_week'])
    task['end_date'] = week_to_date(task['end_week'] + 1)  # End of week
    task['duration_days'] = (task['end_date'] - task['start_date']).days

# Create figure with more bottom margin (smaller size for slide)
fig, ax = plt.subplots(figsize=(10, 4.5))
fig.subplots_adjust(bottom=0.15, top=0.92, left=0.25, right=0.95)

# Set up the plot
y_positions = np.arange(len(tasks))
bar_height = 0.6

# Plot bars using dates
for i, task in enumerate(tasks):
    start = task['start_date']
    duration = task['duration_days']
    ax.barh(i, duration, left=start, height=bar_height, 
            color=task['color'], edgecolor='black', linewidth=1.5, alpha=0.8)

# Set y-axis labels with task names
ax.set_yticks(y_positions)
ax.set_yticklabels([task['name'] for task in tasks], fontsize=10, fontweight='bold')
ax.invert_yaxis()  # Top task at top

# Set x-axis to dates
ax.set_xlabel('Date', fontsize=12, fontweight='bold', labelpad=10)
ax.set_title('Project Schedule: Work Plan and Progress', fontsize=14, fontweight='bold', pad=15)

# Format x-axis as dates
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(MonthLocator())  # Every month
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)

# Set x-axis limits (from week 1 to week 21)
min_date = week_to_date(1)
max_date = week_to_date(21)
ax.set_xlim(min_date, max_date)

# Add grid
ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)

# Add vertical line at current week (week 4 completed)
current_date = week_to_date(4.5)
ax.axvline(x=current_date, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.text(current_date, len(tasks)-0.2, 'Current', rotation=90, ha='right', va='bottom', 
        fontsize=9, color='red', fontweight='bold')

# Add legend at top right, avoiding overlap with bars
completed_patch = mpatches.Patch(color='#2ecc71', label='Completed', alpha=0.8)
planned_patch = mpatches.Patch(color='#3498db', label='Planned', alpha=0.8)
ax.legend(handles=[completed_patch, planned_patch], loc='upper left', fontsize=10, 
          framealpha=0.9, bbox_to_anchor=(0.02, 0.98))

# Adjust layout
plt.tight_layout()

# Save the figure
output_path = 'project_schedule_gantt.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Gantt chart saved to {output_path}")

# Also save as PDF for LaTeX
output_path_pdf = 'project_schedule_gantt.pdf'
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"Gantt chart saved to {output_path_pdf}")

plt.close()
