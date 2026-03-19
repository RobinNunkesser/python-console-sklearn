from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

from imodels import GreedyTreeClassifier, HSTreeClassifier


def _get_tree_estimator(model):
    """Return a fitted sklearn-style tree estimator from the imodels wrapper if available."""
    for attr in ("estimator_", "model_", "best_estimator_"):
        est = getattr(model, attr, None)
        if est is not None and hasattr(est, "tree_"):
            return est
    if hasattr(model, "tree_"):
        return model
    return None


def _abbreviate_feature(name: str) -> str:
    """Convert e.g. 'petal length (cm)' -> 'p<sub>l</sub>'.

    Takes the first character of the first two non-parenthesised words and
    wraps the second in an HTML subscript tag so the DOT label renders
    compactly (e.g. p_l for petal length).
    """
    words = [w for w in name.split() if "(" not in w]
    if len(words) >= 2:
        return f"{words[0][0]}<sub>{words[1][0]}</sub>"
    return name[:4]


def export_compact_dot(estimator, feature_names, class_names) -> str:
    """Build a compact Graphviz DOT string for a fitted sklearn DecisionTree.

    * Inner nodes show only the split condition using abbreviated feature names.
    * Leaf nodes show only the class-probability vector rounded to 2 dp
      (integer when exact, e.g. ``(1,0,0)``).
    * True (<=) branches are solid lines, False (>) branches are dashed.
    """
    tree = estimator.tree_
    LEAF = -1  # sklearn stores TREE_LEAF = -1 in children_left for leaves

    abbrevs = [_abbreviate_feature(f) for f in feature_names]

    def _fmt_leaf(node_id: int) -> str:
        vals = tree.value[node_id][0]
        total = vals.sum()
        fracs = vals / total if total > 0 else vals
        parts = []
        for v in fracs:
            if abs(v - round(v)) < 0.005:
                parts.append(str(int(round(v))))
            else:
                parts.append(f"{v:.2f}")
        return f"({','.join(parts)})"

    lines = [
        "digraph Tree {",
        'node [shape=plain, color="black", fontname="helvetica"] ;',
        'edge [fontname="helvetica"] ;',
    ]

    def _walk(node_id: int) -> None:
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]

        if left == LEAF:
            label = _fmt_leaf(node_id)
        else:
            feat = abbrevs[tree.feature[node_id]]
            thresh = tree.threshold[node_id]
            label = f"{feat} &#8804; {thresh:.2f}"

        lines.append(f"{node_id} [label=<{label}>] ;")

        if left != LEAF:
            lines.append(
                f'{node_id} -> {left} [labeldistance=2.5, labelangle=45, style="solid"] ;'
            )
            _walk(left)
            lines.append(
                f'{node_id} -> {right} [labeldistance=2.5, labelangle=-45, style="dashed"] ;'
            )
            _walk(right)

    _walk(0)
    lines.append("}")
    return "\n".join(lines)


def save_tree_visualizations(model, feature_names, class_names, file_prefix, out_dir="outputs"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    estimator = _get_tree_estimator(model)
    if estimator is None:
        print("No tree estimator found on the fitted model. Skipping visualization export.")
        return

    # Matplotlib PNG – compact: no impurity, no sample counts, proportions only.
    plt.figure(figsize=(10, 6), dpi=150)
    plot_tree(
        estimator,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
        fontsize=8,
    )
    png_path = out_path / f"{file_prefix}_iris_matplotlib.png"
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()
    print(f"Saved tree image (matplotlib): {png_path}")

    # Compact DOT export.
    dot_content = export_compact_dot(estimator, feature_names, class_names)
    dot_path = out_path / f"{file_prefix}_iris_compact.dot"
    dot_path.write_text(dot_content)
    print(f"Saved compact Graphviz DOT file: {dot_path}")

    # Render DOT to PNG when the graphviz Python package and binary are available.
    try:
        from graphviz import Source

        source = Source(dot_content)
        rendered = source.render(
            filename=f"{file_prefix}_iris_compact",
            directory=str(out_path),
            format="png",
            cleanup=True,
        )
        print(f"Saved compact tree image (Graphviz): {rendered}")
    except Exception as exc:
        print(
            "Graphviz PNG render skipped. "
            "Install `graphviz` Python package and Graphviz binary to enable it. "
            f"Reason: {exc}"
        )


def main():
    # Prepare iris data (multiclass classification with 3 classes).
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = list(iris.feature_names)
    class_names = list(iris.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    models = [
        (
            "hstree",
            HSTreeClassifier(random_state=42),
        ),
        (
            "greedytree",
            GreedyTreeClassifier(random_state=42),
        ),
    ]

    for model_name, model in models:
        model.fit(X_train, y_train, feature_names=feature_names)

        preds = model.predict(X_test)
        preds_proba = model.predict_proba(X_test)

        print(f"\n=== {model_name} ===")
        print(model)
        print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
        print(f"F1 (macro): {f1_score(y_test, preds, average='macro'):.4f}")
        print(f"Predicted probabilities shape: {preds_proba.shape}")

        save_tree_visualizations(
            model,
            feature_names,
            class_names,
            file_prefix=model_name,
            out_dir="outputs",
        )


if __name__ == "__main__":
    main()

