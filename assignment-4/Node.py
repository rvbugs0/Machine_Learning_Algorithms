import numpy as np


class DecisionTree:
    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        predictions = []
        for sample in X:
            node = self.tree
            while node['split_feature']:
                if sample[node['split_feature']] <= node['split_value']:
                    node = node['left']
                else:
                    node = node['right']
            predictions.append(node['class'])
        return np.array(predictions)


    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criterion
        if depth == self.max_depth or n_samples < self.min_samples_split or n_classes == 1:
            leaf_value = self._leaf_value(y)
            return {'class': leaf_value}
        
        # Splitting criterion
        feature_idxs = np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)
        best_feature, best_value = self._best_split(X, y, feature_idxs)
        left_idxs = np.argwhere(X[:, best_feature] <= best_value).flatten()
        right_idxs = np.argwhere(X[:, best_feature] > best_value).flatten()
        
        # Stopping criterion
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            leaf_value = self._leaf_value(y)
            return {'class': leaf_value}
        
        # Recursive splitting
        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth=depth+1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth=depth+1)
        
        return {'split_feature': best_feature, 'split_value': best_value, 'left': left, 'right': right}



    def _best_split(self, X, y, feature_idxs):
        if self.criterion == "gini":
            return self._gini_split(X, y, feature_idxs)
        elif self.criterion == "entropy":
            return self._entropy_split(X, y, feature_idxs)
        else:
            return self._misclassification_split(X, y, feature_idxs)

    def _gini_split(self, X, y, feature_idxs):
        best_gini = np.inf
        best_feature = -1
        best_value = -1
        for feature_idx in feature_idxs:
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_idxs = np.argwhere(
                    X[:, feature_idx] <= threshold).flatten()
                right_idxs = np.argwhere(
                    X[:, feature_idx] > threshold).flatten()
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue
                gini = (len(left_idxs)/len(y))*self._gini(y[left_idxs]) + (
                    len(right_idxs)/len(y))*self._gini(y[right_idxs])
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_idx
                    best_value = threshold
        return best_feature, best_value

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        return 1 - np.sum(proportions**2)

    def _leaf_value(self, y):
        _, counts = np.unique(y, return_counts=True)
        return np.argmax(counts)


    def _misclassification_split(self, X, y, feature_idxs):
        best_error = np.inf
        for feature_idx in feature_idxs:
            values = np.unique(X[:, feature_idx])
            for value in values:
                left_idx = X[:, feature_idx] <= value
                y_left = y[left_idx]
                left_counts = np.bincount(y_left, minlength=self.n_classes)

                right_idx = X[:, feature_idx] > value
                y_right = y[right_idx]
                right_counts = np.bincount(y_right, minlength=self.n_classes)

                left_size = len(y_left)
                right_size = len(y_right)

                left_error = 1 - np.max(left_counts) / left_size
                right_error = 1 - np.max(right_counts) / right_size

                error = (left_error * left_size + right_error * right_size) / (left_size + right_size)
                if error < best_error:
                    best_error = error
                    best_feature = feature_idx
                    best_value = value
        return best_feature, best_value

    def _entropy_split(self, X, y, feature_idx, value):
        left_idx = X[:, feature_idx] <= value
        y_left = y[left_idx]
        left_counts = np.bincount(y_left, minlength=self.n_classes)

        right_idx = X[:, feature_idx] > value
        y_right = y[right_idx]
        right_counts = np.bincount(y_right, minlength=self.n_classes)

        left_size = len(y_left)
        right_size = len(y_right)

        left_entropy = self.entropy(left_counts)
        right_entropy = self.entropy(right_counts)

        return (left_entropy * left_size + right_entropy * right_size) / (left_size + right_size)

    def entropy(self, counts):
        """Calculate entropy for an array of counts."""
        counts = counts[counts != 0]  # remove zero counts
        probs = counts / np.sum(counts)
        return -np.sum(probs * np.log2(probs))



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Load the Titanic dataset
titanic_df = pd.read_csv('decision-tree/data/train.csv')

# Preprocess the data
titanic_df = titanic_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
titanic_df = pd.get_dummies(titanic_df, columns=['Sex', 'Embarked'], drop_first=True)
titanic_df = titanic_df.dropna()

X = titanic_df.drop(columns=['Survived']).values
y = titanic_df['Survived'].values

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier and fit it to the training data
dt = DecisionTree(criterion='entropy', max_depth=5, min_samples_split=5, min_samples_leaf=5)
dt.fit(X_train, y_train)

# Predict the labels for the test data and calculate the accuracy
y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test accuracy: {accuracy:.3f}')
