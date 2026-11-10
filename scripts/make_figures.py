"""Draw the training curve in docs/images from results/log_history.json."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e2"
SERIES = [("eval_wer", "WER", "#2a78d6", "o"), ("eval_cer", "CER", "#eb6834", "s")]

history = [e for e in json.loads((ROOT / "results" / "log_history.json").read_text()) if "eval_cer" in e]
steps = [e["step"] for e in history]

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for key, label, color, marker in SERIES:
    values = [e[key] for e in history]
    ax.plot(steps, values, color=color, linewidth=2, marker=marker, markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, solid_capstyle="round")
    ax.annotate(f"{label} {values[0]:.1f}%", (steps[0], values[0]), xytext=(0, 10),
                textcoords="offset points", ha="center", color=INK, fontsize=10)
    ax.annotate(f"{label} {values[-1]:.1f}%", (steps[-1], values[-1]), xytext=(0, 10),
                textcoords="offset points", ha="center", color=INK, fontsize=10)

ax.set_title("Validation error during training (1,000 clips from held-out speakers)",
             loc="left", color=INK, fontsize=12, pad=14)
ax.set_xlabel("Training step", color=MUTED)
ax.set_ylabel("Error rate (%)", color=MUTED)
ax.set_xticks(steps, [f"{s:,}" for s in steps])
ax.set_xlim(500, 6500)
ax.set_ylim(0, 62)
ax.grid(axis="y", color=GRID, linewidth=1)
ax.set_axisbelow(True)
ax.tick_params(colors=MUTED, length=0)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.legend(loc="upper right", frameon=False, labelcolor=INK)

fig.tight_layout()
fig.savefig(ROOT / "docs" / "images" / "training-curve.png", facecolor=SURFACE)
