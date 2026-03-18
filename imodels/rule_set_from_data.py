"""
Generates synthetic CSV data, trains an imodels RuleFit classifier,
and visualizes the learned rules as overlapping rectangles.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from imodels import RuleFitClassifier

RNG = np.random.default_rng(42)
N = 600  # Total number of data points

# ── 1. Generate data ───────────────────────────────────────────────────────
#
# Three regions with defined p(+) values:
#   R1: X1 < 6 and X2 > 6 -> p(+) = 0.8
#   R2: X1 < 5 and X2 < 7 -> p(+) = 0.3
#   R3: X1 > 4 and X2 < 4 -> p(+) = 0.8
#
# Points are drawn uniformly from [0,10]^2; the label depends on
# how many (and which) regions cover the point.

def assign_label(x1, x2, rng):
    in_r1 = (x1 < 6) and (x2 > 6)
    in_r2 = (x1 < 5) and (x2 < 7)
    in_r3 = (x1 > 4) and (x2 < 4)

    # Average probabilities of matching rules (rough approximation)
    probs = []
    if in_r1:
        probs.append(0.8)
    if in_r2:
        probs.append(0.3)
    if in_r3:
        probs.append(0.8)

    if not probs:
        p = 0.15          # no rule region -> rarely positive
    else:
        p = np.mean(probs)

    return int(rng.random() < p)


X1 = RNG.uniform(0, 10, N)
X2 = RNG.uniform(0, 10, N)
y  = np.array([assign_label(x1, x2, RNG) for x1, x2 in zip(X1, X2)])

df = pd.DataFrame({"X1": X1, "X2": X2, "y": y})
csv_path = "rule_set_data.csv"
df.to_csv(csv_path, index=False)
print(f"Data saved: {csv_path} ({N} rows, {y.mean():.2f} positive share)")

# ── 2. Train RuleFit ───────────────────────────────────────────────────────
X = df[["X1", "X2"]].values

model = RuleFitClassifier(
    max_rules=10,
    random_state=42,
)
model.fit(X, y, feature_names=["X1", "X2"])

# Extract rules (columns: rule, coef, support, ...)
rules_df = model.get_rules()
rules_df = rules_df[rules_df["type"] == "rule"].copy()
rules_df = rules_df[rules_df["coef"].abs() > 0.01].copy()
rules_df = rules_df.sort_values("coef", ascending=False).reset_index(drop=True)

print("\nLearned rules:")
print(rules_df[["rule", "coef", "support"]].to_string(index=False))

# ── 3. Helper: rule -> rectangle ──────────────────────────────────────────
def rule_to_rect(rule_str, x_range=(0, 10), y_range=(0, 10)):
    """Parse a RuleFit rule string and return (x0, x1, y0, y1)."""
    import re
    x0, x1 = x_range
    y0, y1 = y_range

    for cond in rule_str.split(" and "):
        cond = cond.strip()
        m = re.match(r"(X1|X2)\s*([<>]=?)\s*([\d.]+)", cond)
        if not m:
            continue
        feat, op, val = m.group(1), m.group(2), float(m.group(3))
        if feat == "X1":
            if ">" in op:
                x0 = max(x0, val)
            else:
                x1 = min(x1, val)
        elif feat == "X2":
            if ">" in op:
                y0 = max(y0, val)
            else:
                y1 = min(y1, val)

    return x0, x1, y0, y1


def coef_to_color(coef):
    return "#6aafd2" if coef > 0 else "#f5b96e"


# ── 4. Visualization ──────────────────────────────────────────────────────
ALPHA = 0.60
TOP_N = 4   # max number of rules to show

display_rules = rules_df.head(TOP_N)

fig = plt.figure(figsize=(6, 8), facecolor="white")
gs  = gridspec.GridSpec(2, 1, height_ratios=[1, 1.6], hspace=0.05)

# Rule text
ax_text = fig.add_subplot(gs[0])
ax_text.axis("off")

for i, (_, row) in enumerate(display_rules.iterrows()):
    y_pos = 0.92 - i * (0.88 / TOP_N)
    # Format rule string for readability
    rule_display = row["rule"].replace(" and ", r" and $\,$")
    rule_display = rule_display.replace("X1", r"$X_1$").replace("X2", r"$X_2$")
    prob_val = model.predict_proba(
        np.array([[5, 5]])  # Placeholder value - used to show coef sign
    )[0][1]
    sign = "+" if row["coef"] > 0 else "−"
    ax_text.text(0.02, y_pos,
                 f"IF {rule_display}:",
                 transform=ax_text.transAxes, fontsize=11, va="top")
    ax_text.text(0.74, y_pos,
                 f"coef = {row['coef']:+.2f}",
                 transform=ax_text.transAxes, fontsize=11, va="top")

# Rectangles
ax = fig.add_subplot(gs[1])
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
ax.set_xlabel(r"$X_1$", fontsize=13)
ax.set_ylabel(r"$X_2$", fontsize=13)
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

for _, row in display_rules.iterrows():
    x0, x1, y0, y1 = rule_to_rect(row["rule"])
    if x1 <= x0 or y1 <= y0:
        continue
    color = coef_to_color(row["coef"])
    rect  = mpatches.FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="square,pad=0",
        linewidth=0.8, edgecolor="#555555",
        facecolor=color, alpha=ALPHA
    )
    ax.add_patch(rect)

out_path = "rule_set_from_data.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
print(f"\nVisualization saved: {out_path}")
