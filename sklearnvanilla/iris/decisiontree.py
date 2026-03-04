from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y)

# Plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree Classifier on Iris Dataset")
plt.tight_layout()
plt.show()

# Print accuracy
accuracy = clf.score(X, y)
print(f"Accuracy: {accuracy:.4f}")