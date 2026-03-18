"""
Visualization of a rule set with matplotlib.
Shows three rules with overlapping rectangles in the X1-X2 space.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

# --- Rule set ---
rules = [
    {"label": r"IF $X_1$< 6 and $X_2$> 6:", "prob": "p(+) = 0.8",
     "x": (0, 6), "y": (6, 10), "p": 0.8},
    {"label": r"IF $X_1$< 5 and $X_2$< 7:", "prob": "p(+) = 0.3",
     "x": (0, 5), "y": (0, 7), "p": 0.3},
    {"label": r"IF $X_1$> 4 and $X_2$< 4:", "prob": "p(+) = 0.8",
     "x": (4, 10), "y": (0, 4), "p": 0.8},
]

# Color map: high probability -> blue, low probability -> orange
def prob_to_color(p):
    if p >= 0.6:
        return "#6aafd2"   # blue
    else:
        return "#f5b96e"   # orange

ALPHA = 0.65
XMAX, YMAX = 10, 10

fig = plt.figure(figsize=(6, 8), facecolor="white")
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.6], hspace=0.05)

# ── Top area: rule text ───────────────────────────────────────────────────
ax_text = fig.add_subplot(gs[0])
ax_text.axis("off")

for i, rule in enumerate(rules):
    y_pos = 0.85 - i * 0.32
    ax_text.text(0.02, y_pos, rule["label"],
                 transform=ax_text.transAxes,
                 fontsize=13, va="top",
                 fontfamily="sans-serif")
    ax_text.text(0.72, y_pos, rule["prob"],
                 transform=ax_text.transAxes,
                 fontsize=13, va="top",
                 fontfamily="sans-serif")

# ── Bottom area: rectangle visualization ──────────────────────────────────
ax = fig.add_subplot(gs[1])
ax.set_xlim(0, XMAX)
ax.set_ylim(0, YMAX)
ax.set_aspect("equal")
ax.set_xlabel(r"$X_1$", fontsize=13)
ax.set_ylabel(r"$X_2$", fontsize=13)
ax.tick_params(left=False, bottom=False,
               labelleft=False, labelbottom=False)

# Axis frame: bottom and left only
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# Draw rectangles (order affects visible overlap)
for rule in rules:
    x0, x1 = rule["x"]
    y0, y1 = rule["y"]
    color = prob_to_color(rule["p"])
    rect = mpatches.FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="square,pad=0",
        linewidth=0.8, edgecolor="#555555",
        facecolor=color, alpha=ALPHA
    )
    ax.add_patch(rect)

plt.savefig("rule_set_visualization.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.show()
print("Saved: rule_set_visualization.png")
