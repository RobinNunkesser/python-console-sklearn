"""Create a three-panel rule visualization similar to the reference image.

This script intentionally omits the "Algebraic models" panel.
"""

from matplotlib import patches
import matplotlib.pyplot as plt


def setup_panel(ax, title):
	"""Configure one outer panel with light background and title."""
	ax.set_facecolor("#f3f3f3")
	ax.set_xticks([])
	ax.set_yticks([])
	for spine in ax.spines.values():
		spine.set_visible(True)
		spine.set_color("#d0d3d6")
		spine.set_linewidth(1.0)

	# Header divider line
	ax.plot([0, 1], [0.85, 0.85], color="#d0d3d6", lw=1.0, transform=ax.transAxes)
	ax.text(
		0.5,
		0.92,
		title,
		transform=ax.transAxes,
		ha="center",
		va="center",
		fontsize=22,
		fontweight="bold",
		color="#222222",
	)


def draw_rule_set(ax):
	"""Draw rule set text and overlapping rectangles."""
	text_y = [0.77, 0.69, 0.61]
	left = [
		r"IF $X_1$<6 and $X_2$>6:",
		r"IF $X_1$<5 and $X_2$<7:",
		r"IF $X_1$>4 and $X_2$<4:",
	]
	right = ["p(+) = 0.8", "p(+) = 0.3", "p(+) = 0.8"]
	for y, ltxt, rtxt in zip(text_y, left, right):
		ax.text(0.07, y, ltxt, transform=ax.transAxes, fontsize=20, fontweight="bold")
		ax.text(0.62, y, rtxt, transform=ax.transAxes, fontsize=20)

	h = ax.inset_axes([0.14, 0.10, 0.72, 0.45])
	h.set_xlim(0, 10)
	h.set_ylim(0, 10)
	h.set_xticks([])
	h.set_yticks([])
	h.spines["top"].set_visible(False)
	h.spines["right"].set_visible(False)
	h.spines["left"].set_color("#8b8b8b")
	h.spines["bottom"].set_color("#8b8b8b")
	h.spines["left"].set_linewidth(1.8)
	h.spines["bottom"].set_linewidth(1.8)

	blue = "#7ea7cf"
	orange = "#e7b472"
	edge = "#4b8ec6"

	h.add_patch(patches.Rectangle((0.5, 5.6), 5.4, 3.7, facecolor=blue, edgecolor=edge, lw=1.5, alpha=0.8))
	h.add_patch(patches.Rectangle((0.5, 0.5), 4.3, 6.5, facecolor=orange, edgecolor="#b77d3e", lw=1.5, alpha=0.8))
	h.add_patch(patches.Rectangle((4.1, 0.5), 5.5, 4.0, facecolor=blue, edgecolor=edge, lw=1.5, alpha=0.8))

	h.set_xlabel(r"$X_1$", fontsize=24, labelpad=12)
	h.set_ylabel(r"$X_2$", fontsize=24, labelpad=10, rotation=0)
	h.yaxis.set_label_coords(-0.10, 0.5)


def draw_rule_list(ax):
	"""Draw rule list text and block-partition rectangles."""
	rows = [
		(r"IF $X_1$<5:", "p(+) = 0.3"),
		(r"ELSE If $X_2$<4:", "p(+) = 0.9"),
		(r"ELSE If $X_1$>6:", "p(+) = 0.7"),
		("ELSE", "p(+) = 0.5"),
	]
	y = 0.77
	for ltxt, rtxt in rows:
		ax.text(0.07, y, ltxt, transform=ax.transAxes, fontsize=20, fontweight="bold")
		ax.text(0.64, y, rtxt, transform=ax.transAxes, fontsize=20)
		y -= 0.06

	h = ax.inset_axes([0.14, 0.10, 0.72, 0.45])
	h.set_xlim(0, 10)
	h.set_ylim(0, 10)
	h.set_xticks([])
	h.set_yticks([])
	h.spines["top"].set_visible(False)
	h.spines["right"].set_visible(False)
	h.spines["left"].set_color("#8b8b8b")
	h.spines["bottom"].set_color("#8b8b8b")
	h.spines["left"].set_linewidth(1.8)
	h.spines["bottom"].set_linewidth(1.8)

	blue = "#7ea7cf"
	orange = "#e7b472"
	gray = "#d8d8d8"
	edge = "#4b8ec6"

	h.add_patch(patches.Rectangle((0.5, 0.5), 4.6, 8.8, facecolor=orange, edgecolor="#b77d3e", lw=1.4, alpha=0.85))
	h.add_patch(patches.Rectangle((4.9, 0.5), 5.0, 3.4, facecolor=blue, edgecolor=edge, lw=1.4, alpha=0.75))
	h.add_patch(patches.Rectangle((4.9, 3.9), 5.0, 5.4, facecolor=gray, edgecolor=edge, lw=1.4, alpha=0.55))
	h.add_patch(patches.Rectangle((4.9, 3.9), 1.2, 5.4, facecolor=blue, edgecolor=edge, lw=1.4, alpha=0.35))

	h.set_xlabel(r"$X_1$", fontsize=24, labelpad=12)
	h.set_ylabel(r"$X_2$", fontsize=24, labelpad=10, rotation=0)
	h.yaxis.set_label_coords(-0.10, 0.5)


def draw_rule_tree(ax):
	"""Draw rule tree text diagram and piecewise rectangles."""
	# Tiny tree diagram in upper area
	tree_ax = ax.inset_axes([0.08, 0.56, 0.84, 0.27])
	tree_ax.set_axis_off()

	# Labels
	tree_ax.text(0.50, 0.95, r"IF $X_1$>5", ha="center", va="center", fontsize=24, fontweight="bold")
	tree_ax.text(0.25, 0.63, r"IF $X_2$>6", ha="center", va="center", fontsize=24, fontweight="bold")
	tree_ax.text(0.75, 0.63, r"IF $X_2$>4", ha="center", va="center", fontsize=24, fontweight="bold")

	tree_ax.text(0.12, 0.22, "0.2", ha="center", va="center", fontsize=22)
	tree_ax.text(0.38, 0.22, "0.9", ha="center", va="center", fontsize=22)
	tree_ax.text(0.62, 0.22, "0.8", ha="center", va="center", fontsize=22)
	tree_ax.text(0.88, 0.22, "0.6", ha="center", va="center", fontsize=22)
	tree_ax.text(0.02, 0.22, "p(+):", ha="right", va="center", fontsize=24)

	# Branch lines
	line = {"color": "#8c8c8c", "lw": 2.0}
	tree_ax.plot([0.50, 0.30], [0.87, 0.70], **line)
	tree_ax.plot([0.50, 0.70], [0.87, 0.70], **line)
	tree_ax.plot([0.25, 0.12], [0.56, 0.34], **line)
	tree_ax.plot([0.25, 0.38], [0.56, 0.34], **line)
	tree_ax.plot([0.75, 0.62], [0.56, 0.34], **line)
	tree_ax.plot([0.75, 0.88], [0.56, 0.34], **line)

	h = ax.inset_axes([0.14, 0.10, 0.72, 0.45])
	h.set_xlim(0, 10)
	h.set_ylim(0, 10)
	h.set_xticks([])
	h.set_yticks([])
	h.spines["top"].set_visible(False)
	h.spines["right"].set_visible(False)
	h.spines["left"].set_color("#8b8b8b")
	h.spines["bottom"].set_color("#8b8b8b")
	h.spines["left"].set_linewidth(1.8)
	h.spines["bottom"].set_linewidth(1.8)

	blue = "#7ea7cf"
	orange = "#e7b472"
	gray = "#d8d8d8"
	edge = "#4b8ec6"

	h.add_patch(patches.Rectangle((0.5, 6.5), 4.4, 2.8, facecolor=blue, edgecolor=edge, lw=1.5, alpha=0.82))
	h.add_patch(patches.Rectangle((0.5, 0.5), 4.4, 5.9, facecolor=orange, edgecolor="#b77d3e", lw=1.5, alpha=0.85))
	h.add_patch(patches.Rectangle((4.9, 3.9), 5.0, 5.4, facecolor=gray, edgecolor=edge, lw=1.5, alpha=0.55))
	h.add_patch(patches.Rectangle((4.9, 0.5), 5.0, 3.4, facecolor=blue, edgecolor=edge, lw=1.5, alpha=0.75))

	h.set_xlabel(r"$X_1$", fontsize=24, labelpad=12)
	h.set_ylabel(r"$X_2$", fontsize=24, labelpad=10, rotation=0)
	h.yaxis.set_label_coords(-0.10, 0.5)


def main():
	fig, axes = plt.subplots(1, 3, figsize=(22, 8), constrained_layout=True)
	setup_panel(axes[0], "Rule set")
	setup_panel(axes[1], "Rule list")
	setup_panel(axes[2], "Rule tree")

	draw_rule_set(axes[0])
	draw_rule_list(axes[1])
	draw_rule_tree(axes[2])

	out = "three_rule_models.png"
	fig.savefig(out, dpi=180, facecolor="white")
	plt.show()
	print(f"Saved {out}")


if __name__ == "__main__":
	main()
