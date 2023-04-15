import numpy as np
from collections import Counter


class RandomForest:
    def __init__(self, classifier, num_trees=10, min_features=None):
        self.classifier = classifier
        self.num_trees = num_trees
        self.min_features = min_features
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.num_trees):
            tree = 
            if self.min_features is not None:
                # Randomly select a subset of features
                feature_indices = np.random.choice(X.shape[1], self.min_features, replace=False)
                tree.fit(X[:, feature_indices], y)
            else:
                tree.fit(X, y)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Majority vote for classification
        if self.classifier:
            return np.round(np.mean(predictions, axis=0))
        # Average for regression
        else:
            return np.mean(predictions, axis=0)
