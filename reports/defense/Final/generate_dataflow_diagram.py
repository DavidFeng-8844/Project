"""
Generate the System Data Flow Diagram (Figure 2) for the final report.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Colors ────────────────────────────────────
BG        = "#FAFBFC"
C_USER    = "#2563EB"
C_BACK    = "#7C3AED"
C_DB      = "#059669"
C_EXEC    = "#D97706"
C_REG     = "#DC2626"
C_FRONT   = "#0891B2"
C_ARROW   = "#374151"
C_LABEL   = "#111827"
C_NOTE    = "#6B7280"
C_WHITE   = "#FFFFFF"

F_TITLE = {"family": "sans-serif", "weight": "bold", "size": 14}
F_BOX   = {"family": "sans-serif", "weight": "bold", "size": 9}
F_SUB   = {"family": "sans-serif", "size": 7}
F_ARR   = {"family": "sans-serif", "size": 7, "style": "italic"}
F_NOTE  = {"family": "sans-serif", "size": 6, "color": C_NOTE}
F_STEP  = {"family": "sans-serif", "weight": "bold", "size": 7}


def rbox(ax, cx, cy, w, h, color, title, sub=None):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.015", fc=color, ec="white", lw=1.2, zorder=3)
    ax.add_patch(box)
    if sub:
        ax.text(cx, cy + h * 0.15, title, ha="center", va="center",
                color=C_WHITE, fontdict=F_BOX, zorder=4)
        ax.text(cx, cy - h * 0.2, sub, ha="center", va="center",
                color="#D1D5DB", fontdict=F_SUB, zorder=4)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                color=C_WHITE, fontdict=F_BOX, zorder=4)


def arr(ax, x1, y1, x2, y2, label=None, lo=(0, 0), color=C_ARROW):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="-|>", connectionstyle="arc3,rad=0",
                        color=color, lw=1.2, mutation_scale=11, zorder=2)
    ax.add_patch(a)
    if label:
        mx = (x1 + x2) / 2 + lo[0]
        my = (y1 + y2) / 2 + lo[1]
        ax.text(mx, my, label, ha="center", va="bottom",
                fontdict=F_ARR, color=C_ARROW, zorder=5)


def step(ax, x, y, n):
    c = plt.Circle((x, y), 0.018, color="#1F2937", zorder=6)
    ax.add_patch(c)
    ax.text(x, y, str(n), ha="center", va="center",
            color="white", fontdict=F_STEP, zorder=7)


# ── Figure ────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")

# Title
ax.text(5, 5.7, "System Data Flow Diagram", ha="center", va="top",
        fontdict=F_TITLE, color=C_LABEL)
ax.text(5, 5.4, "Request  →  Queue  →  Inference  →  Polling  →  Result",
        ha="center", va="top", fontdict={**F_SUB, "size": 8.5}, color=C_NOTE)

# ── Box dimensions ────────────────────────────
BW, BH = 1.6, 0.7
MW, MH = 2.0, 0.9

# ── Positions ─────────────────────────────────
UX, UY   = 1.5, 4.5   # User
BKX, BKY = 4.5, 4.5   # Backend
FX, FY   = 1.5, 2.8   # Frontend
DX, DY   = 4.5, 2.8   # Database
EX, EY   = 4.5, 1.0   # Executor
RX, RY   = 8.0, 1.0   # Registry

# ── Draw boxes ────────────────────────────────
rbox(ax, UX, UY,  BW, BH, C_USER,  "User (Browser)", "Web Frontend")
rbox(ax, BKX, BKY, BW, BH, C_BACK, "Flask Backend", "REST API")
rbox(ax, FX, FY,  BW, BH, C_FRONT, "Frontend", "Polling Loop (1 s)")
rbox(ax, DX, DY,  BW, BH, C_DB,   "Database", "SQLite · tenant_id")
rbox(ax, EX, EY,  MW, MH, C_EXEC, "ThreadPoolExecutor", "max_workers = 4")
rbox(ax, RX, RY,  BW+0.2, BH+0.1, C_REG, "Model Registry", "bird_nest · oil_leak · infrared")

# Worker sub-boxes
for i, w in enumerate(["W1", "W2", "W3", "W4"]):
    wx = EX - 0.6 + i * 0.4
    wy = EY - 0.22
    sb = FancyBboxPatch((wx - 0.14, wy - 0.1), 0.28, 0.2,
                        boxstyle="round,pad=0.02", fc="#F59E0B",
                        ec="white", lw=0.6, zorder=4)
    ax.add_patch(sb)
    ax.text(wx, wy, w, ha="center", va="center", color="white",
            fontdict={"size": 6, "weight": "bold"}, zorder=5)

# ── Arrows ────────────────────────────────────

# ① User → Backend
step(ax, 3.0, 4.95, 1)
arr(ax, UX + BW/2, UY + 0.12,   BKX - BW/2, BKY + 0.12,
    "① POST /inference/tasks\n(image + model_id)", lo=(0, 0.08))

# ② Backend → User
step(ax, 3.0, 4.0, 2)
arr(ax, BKX - BW/2, BKY - 0.12,  UX + BW/2, UY - 0.12,
    "② HTTP 202 (task_id)", lo=(0, -0.18))

# ③ Backend → DB
step(ax, 5.15, 3.7, 3)
arr(ax, BKX + 0.1, BKY - BH/2,    DX + 0.1, DY + BH/2,
    "③ INSERT task\n(PENDING)", lo=(0.4, 0))

# ④ Frontend → User (polls via backend)
step(ax, 0.6, 3.65, 4)
arr(ax, FX, FY + BH/2,   UX, UY - BH/2,
    "④ GET /tasks/<id>/status", lo=(-0.55, 0))

# ⑤ DB → Frontend
step(ax, 3.0, 2.8, 5)
arr(ax, DX - BW/2, DY,   FX + BW/2, FY,
    "⑤ Return results\n(status = SUCCESS)", lo=(0, 0.12))

# ⑥ DB ↔ Executor
step(ax, 5.15, 2.0, 6)
arr(ax, DX + 0.15, DY - BH/2,    EX + 0.15, EY + MH/2,
    "⑥ Worker picks\nPENDING task", lo=(0.5, 0))

arr(ax, EX - 0.15, EY + MH/2,    DX - 0.15, DY - BH/2,
    "UPDATE → SUCCESS", lo=(-0.55, 0), color=C_DB)

# Executor → Registry
arr(ax, EX + MW/2, EY,   RX - (BW+0.2)/2, RY,
    "Load weights", lo=(0, 0.1))

# ── Notes ─────────────────────────────────────
ax.text(DX, DY - BH/2 - 0.1,
        "All queries filtered by tenant_id → strict data isolation",
        ha="center", va="top", fontdict=F_NOTE)

ax.text(FX, FY - BH/2 - 0.1,
        "Download JSON reports\nfor offline analysis",
        ha="center", va="top", fontdict=F_NOTE)

# ── Legend ────────────────────────────────────
legend = [(C_USER, "User"), (C_BACK, "Backend"), (C_FRONT, "Frontend"),
          (C_DB, "Database"), (C_EXEC, "Thread Pool"), (C_REG, "Model Registry")]
for i, (c, t) in enumerate(legend):
    lx = 0.3 + i * 1.3
    ly = 0.15
    p = FancyBboxPatch((lx, ly), 0.22, 0.14,
                       boxstyle="round,pad=0.01", fc=c, ec="white", lw=0.5, zorder=4)
    ax.add_patch(p)
    ax.text(lx + 0.3, ly + 0.07, t, va="center",
            fontdict={"size": 6.5}, color=C_LABEL, zorder=5)

# ── Save ──────────────────────────────────────
out = "/home/david/srtp/mcnn-pytorch-train/reports/defense/Final/dataflow_diagram.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"✓ Saved to {out}")
