from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz, plot_tree

from imodels import HSTreeClassifier


def _get_tree_estimator(model):
    """Return a fitted sklearn-style tree estimator from the imodels wrapper if available."""
    for attr in ("estimator_", "model_", "best_estimator_"):
        est = getattr(model, attr, None)
        if est is not None and hasattr(est, "tree_"):
            return est
    if hasattr(model, "tree_"):
        return model
    return None


def save_tree_visualizations(model, feature_names, class_names, out_dir="outputs"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    estimator = _get_tree_estimator(model)
    if estimator is None:
        print("No tree estimator found on the fitted model. Skipping visualization export.")
        return

    # Always create a PNG via matplotlib (works without Graphviz installation).
    plt.figure(figsize=(10, 6), dpi=150)
    plot_tree(
        estimator,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=False,
        fontsize=8,
    )
    png_path = out_path / "hstree_iris_matplotlib.png"
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()
    print(f"Saved tree image (matplotlib): {png_path}")

    # Optionally export Graphviz DOT and render to PNG when graphviz Python package and binary exist.
    dot_path = out_path / "hstree_iris_graphviz.dot"
    export_graphviz(
        estimator,
        out_file=str(dot_path),
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True,
    )
    print(f"Saved Graphviz DOT file: {dot_path}")

    try:
        from graphviz import Source

        source = Source.from_file(str(dot_path))
        rendered = source.render(
            filename="hstree_iris_graphviz",
            directory=str(out_path),
            format="png",
            cleanup=True,
        )
        print(f"Saved tree image (Graphviz): {rendered}")
    except Exception as exc:
        print(
            "Graphviz PNG render skipped. Install `graphviz` Python package and Graphviz binary to enable it. "
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

    # HSTreeClassifierCV currently errors on Iris in this environment due to tiny negative proba values.
    # Use HSTreeClassifier for a stable, fully runnable multiclass example.
    model = HSTreeClassifier(max_leaf_nodes=4, reg_param=10, random_state=42)
    model.fit(X_train, y_train, feature_names=feature_names)

    preds = model.predict(X_test)
    preds_proba = model.predict_proba(X_test)

    print(model)
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"F1 (macro): {f1_score(y_test, preds, average='macro'):.4f}")
    print(f"Predicted probabilities shape: {preds_proba.shape}")

    save_tree_visualizations(model, feature_names, class_names, out_dir="outputs")


if __name__ == "__main__":
    main()

