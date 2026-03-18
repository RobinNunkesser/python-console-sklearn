"""
all_three_models.py
-------------------
1. Generates synthetic CSV data (X1, X2, y)
2. Trains Rule set (RuleFit), Rule list (GreedyRuleList),
   Rule tree (GreedyTree) from imodels
3. Visualizes all three models side by side (text + heatmap)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from imodels import RuleFitClassifier, GreedyRuleListClassifier, GreedyTreeClassifier

# ── 1. Synthetic data ─────────────────────────────────────────────────────
# Probability structure follows the original figure:
#   Rule list: IF X1<5 → 0.3 | ELSE IF X2<4 → 0.9 | ELSE IF X1>6 → 0.7 | ELSE 0.5
RNG = np.random.default_rng(42)
N   = 1000

def ground_truth_p(x1, x2):
    if x1 < 5:    return 0.3
    elif x2 < 4:  return 0.9
    elif x1 > 6:  return 0.7
    else:         return 0.5

X1 = RNG.uniform(0, 10, N)
X2 = RNG.uniform(0, 10, N)
y  = np.array([int(RNG.random() < ground_truth_p(a, b)) for a, b in zip(X1, X2)])

df = pd.DataFrame({"X1": X1, "X2": X2, "y": y})
df.to_csv("three_models_data.csv", index=False)
print(f"CSV saved ({N} rows, positive share: {y.mean():.2f})")


X = df[["X1", "X2"]].values

# ── 2. Train models ───────────────────────────────────────────────────────
rule_set  = RuleFitClassifier(max_rules=6, random_state=42)
rule_set.fit(X, y, feature_names=["X1", "X2"])

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

# ── 4. Prepare rule text ──────────────────────────────────────────────────

def xi(s):
    """Replace X1/X2 with LaTeX subscripts."""
    return s.replace("X1", r"$X_1$").replace("X2", r"$X_2$")

# --- Rule set ---------------------------------------------------------------
rs_df = rule_set.get_rules()
rs_df = rs_df[(rs_df["type"] == "rule") & (rs_df["coef"].abs() > 0.01)]
rs_df = rs_df.sort_values("coef", ascending=False).head(4).reset_index(drop=True)
rs_text = []
for _, r in rs_df.iterrows():
    rs_text.append((f"IF {xi(r['rule'])}:", f"c = {r['coef']:+.2f}"))

# --- Rule list ---------------------------------------------------------------
rl_text = []
for i, rule in enumerate(rule_list.rules_):
    if "col" in rule:
        prefix = "IF" if rule["depth"] == 0 else "ELSE IF"
        sign   = "<" if rule["flip"] else ">"
        cond   = f"{xi(rule['col'])} {sign} {rule['cutoff']:.1f}"
        prob   = rule["val_right"]
        rl_text.append((f"{prefix} {cond}:", f"p(+) = {prob:.2f}"))
    else:
        rl_text.append(("ELSE:", f"p(+) = {rule['val']:.2f}"))

# ── 5. Draw tree diagram ──────────────────────────────────────────────────

def draw_tree(ax, model):
    """Draw a depth-2 tree diagram in ax (GreedyTreeClassifier)."""
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")

    t  = model.tree_
    fn = model.feature_names_in_ if hasattr(model, "feature_names_in_") else ["X1", "X2"]

    def leaf_prob(n):
        v = t.value[n][0]
        return v[1] / v.sum()

    def split_label(n):
        f   = fn[t.feature[n]]
        thr = t.threshold[n]
        return f"IF {xi(f)} > {thr:.1f}"

    NODE_KW  = dict(ha="center", va="center", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="#888", lw=0.8))
    LEAF_KW  = dict(ha="center", va="center", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.25", fc="#dde8f5",
                              ec="#888", lw=0.8))

    def add_edge(p, c):
        ax.annotate("", xy=c, xytext=p,
                    arrowprops=dict(arrowstyle="-", color="#777", lw=0.9))

    # Positions: root, L, R, LL, LR, RL, RR
    pos = {0: (0.50, 0.88),
           1: (0.25, 0.57),   # children_left[0]
           2: (0.75, 0.57),   # children_right[0]
           3: (0.12, 0.18),
           4: (0.38, 0.18),
           5: (0.62, 0.18),
           6: (0.88, 0.18)}

    node_ids = [
        0,
        t.children_left[0],
        t.children_right[0],
    ]
    for child_idx, parent in [(1, 0), (2, 0)]:
        nid = node_ids[child_idx]
        if t.children_left[nid] != -1:
            node_ids += [t.children_left[nid], t.children_right[nid]]

    def draw_node(nid, xy, is_leaf):
        if is_leaf:
            p = leaf_prob(nid)
            ax.text(*xy, f"p(+)={p:.2f}", **LEAF_KW)
        else:
            ax.text(*xy, split_label(nid), **NODE_KW)

    # Root
    draw_node(node_ids[0], pos[0], is_leaf=False)

    leaf_counter = 3
    for child_slot, parent_pos in [(1, pos[0]), (2, pos[0])]:
        if child_slot >= len(node_ids):
            break
        nid    = node_ids[child_slot]
        c_pos  = pos[child_slot]
        add_edge(parent_pos, c_pos)
        is_leaf = t.children_left[nid] == -1
        draw_node(nid, c_pos, is_leaf)

        if not is_leaf:
            for grandchild in [t.children_left[nid], t.children_right[nid]]:
                gc_pos = pos[leaf_counter]
                add_edge(c_pos, gc_pos)
                draw_node(grandchild, gc_pos, is_leaf=True)
                leaf_counter += 1

# ── 6. Build figure ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 9), facecolor="white")
outer = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32)

panels = [
    ("Rule set",  "set",  "text", rs_text),
    ("Rule list", "list", "text", rl_text),
    ("Rule tree", "tree", "tree", None),
]

for col, (title, key, mode, text_lines) in enumerate(panels):
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[col],
        height_ratios=[1.2, 1.4], hspace=0.06)

    # ── Top area (text or tree) ───────────────────────────────────────────
    ax_top = fig.add_subplot(inner[0])
    ax_top.set_title(title, fontsize=14, fontweight="bold", pad=8)

    if mode == "text":
        ax_top.axis("off")
        step = 0.88 / max(len(text_lines), 1)
        for i, (left, right) in enumerate(text_lines):
            yp = 0.97 - i * step
            ax_top.text(0.02, yp, left,  transform=ax_top.transAxes,
                        fontsize=9.5, va="top")
            ax_top.text(0.74, yp, right, transform=ax_top.transAxes,
                        fontsize=9.5, va="top")
    else:
        draw_tree(ax_top, rule_tree)

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
