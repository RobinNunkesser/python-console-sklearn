
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from imodels import RuleFitClassifier, GreedyRuleListClassifier, GreedyTreeClassifier, SlipperClassifier

# ── 1. Synthetic data ─────────────────────────────────────────────────────
# Probability structure follows the original figure:
#   Rule list: IF X1<5 → 0.3 | ELSE IF X2<4 → 0.9 | ELSE IF X1>6 → 0.7 | ELSE 0.5
RNG = np.random.default_rng(42)
N   = 1000

def ground_truth_rl(x1, x2):
    if x1 < 5:    return 0.3
    elif x2 < 4:  return 0.9
    elif x1 > 6:  return 0.7
    else:         return 0.5

def ground_truth_p(x1, x2):
    return 0.8 * (x1 < 6) * (x2 > 6) + 0.3 * (x1 < 5) * (x2 < 7) + 0.8 * (x1 > 4) * (x2 < 4)

X1 = RNG.uniform(0, 10, N)
X2 = RNG.uniform(0, 10, N)
y  = np.array([int(RNG.random() < ground_truth_p(a, b)) for a, b in zip(X1, X2)])
#y[RNG.choice(len(y), size=len(y)//3, replace=False)] = 2

df = pd.DataFrame({"X1": X1, "X2": X2, "y": y})
df.to_csv("three_models_data.csv", index=False)
print(f"CSV saved ({N} rows, positive share: {y.mean():.2f})")

X = df[["X1", "X2"]].values

# ── 2. Train models ───────────────────────────────────────────────────────
rule_set = SlipperClassifier()
#rule_set  = RuleFitClassifier(max_rules=4, random_state=42)
rule_set.fit(X, y, feature_names=["X1", "X2"])
print(rule_set)  # Shows rule text in the console
rule_set._rules_df  # Shows rules as a DataFrame in the console
print(rule_set.predict_proba(np.array([[1, 1], [5.5, 1], [6.5, 1],[1,6.5],[1,7.5]])))  # Shows predicted probabilities for sample points
print(rule_set.predict(np.array([[1, 1], [5.5, 1], [6.5, 1],[1,6.5],[1,7.5]])))  # Shows predicted classes for sample points

rule_list = GreedyRuleListClassifier(max_depth=4)
rule_list.fit(X, y, feature_names=["X1", "X2"])

rule_tree = GreedyTreeClassifier(max_depth=2)
rule_tree.fit(X, y, feature_names=["X1", "X2"])

print("Models trained.")
print(rule_list)  # Shows rule text in the console

# ── 3. Heatmaps ───────────────────────────────────────────────────────────
gx, gy = np.meshgrid(np.linspace(0, 10, 300), np.linspace(0, 10, 300))
grid   = np.c_[gx.ravel(), gy.ravel()]

Z = {
    "set":  rule_set.predict_proba(grid)[:, 1].reshape(gx.shape),
    "list": rule_list.predict_proba(grid)[:, 1].reshape(gx.shape),
    "tree": rule_tree.predict_proba(grid)[:, 1].reshape(gx.shape),
}

# orange → white → blue (low → medium → high probability)
cmap = mcolors.LinearSegmentedColormap.from_list(
    "ot_b", ["#f5b96e", "#eeeeee", "#6aafd2"])

# ── 6. Build figure ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 9), facecolor="white")
outer = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32)

panels = [
    ("Rule set",  "set",  "text", "Lorem ipsum"),
    ("Rule list", "list", "text", "Lorem ipsum"),
    ("Rule tree", "tree", "text", "Lorem ipsum"),
]

for col, (title, key, mode, text_lines) in enumerate(panels):
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[col],
        height_ratios=[1.2, 1.4], hspace=0.06)

    # ── Top area (text or tree) ───────────────────────────────────────────
    ax_top = fig.add_subplot(inner[0])
    ax_top.set_title(title, fontsize=14, fontweight="bold", pad=8)

    

    # ── Bottom area (heatmap) ─────────────────────────────────────────────
    ax_h = fig.add_subplot(inner[1])
    ax_h.pcolormesh(gx, gy, Z[key], cmap=cmap, vmin=0, vmax=1, shading="auto")
    ax_h.set_xlim(0, 10)
    ax_h.set_ylim(0, 10)
    ax_h.set_xticks([])
    ax_h.set_yticks([])
    ax_h.set_xlabel(r"$X_1$", fontsize=11)
    ax_h.set_ylabel(r"$X_2$", fontsize=11)
    for spine in ["top", "right"]:
        ax_h.spines[spine].set_visible(False)

out_path = "three_models_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
print(f"Saved: {out_path}")    
