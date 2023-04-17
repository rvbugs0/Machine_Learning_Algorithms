import numpy as np
from collections import Counter
import copy
import random
import pandas as pd
from DecisionTree import decision_tree_classifier
from DecisionTree import Criterion
from AdaBoostClassifier import AdaBoostClassifier
random.seed(None)


class RandomForest:
    def __init__(self, classifier, num_trees=10, min_features=None):
        self.classifier = classifier
        self.num_trees = num_trees
        self.min_features = min_features
        self.trees = []

    def fit(self, X, y):
        num_columns = len(X.columns)
        if (num_columns < self.min_features):
            print("Minimum number of features is set to a number greater than the total no. of features in the supplied dataframe")
            return
        
        features = X.columns.to_numpy().tolist()
        # print(features)

        for _ in range(self.num_trees):
            random.seed(None)
            if(type(self.classifier)==decision_tree_classifier):
                tree = decision_tree_classifier(self.classifier.criterion,self.classifier.min_samples_split,self.classifier.max_depth,self.classifier.min_samples_leaf)
            else:
                tree = AdaBoostClassifier(num_learners=self.classifier.num_learners,learning_rate=self.classifier.learning_rate)
            random.shuffle(features)
            random_features = random.sample(features, random.randint(self.min_features, len(features)))
            print(random_features)
            if self.min_features is not None:
                input = X[np.array(random_features)]
                input = input.sample(n=X.shape[0], replace=True)
                tree.fit(input, y)
            else:
                tree.fit(X, y)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.round(np.mean(predictions, axis=0))


if __name__ == '__main__':
    # Reading data
    d = pd.read_csv(
        "data/train.csv")[['Age', 'Sex', 'Fare', 'Pclass', 'Survived']].dropna()
    d = d.assign(Sex=d.Sex.eq('male').astype(int))

    # Constructing the X and Y matrices
    X = d[['Age', 'Sex', 'Fare', 'Pclass']]
    Y = d['Survived'].values.tolist()

    # Initiating the Node
    # print("USING DECISION TREE")

    # dt = decision_tree_classifier(Criterion.GINI_IMPURITY, min_samples_split=12, max_depth=5, min_samples_leaf=5)
    dt =     AdaBoostClassifier(10)
    rf = RandomForest(dt,5,1)
    rf.fit(X,Y)


    # Predicting
    Xsubset = X.copy()
    Xsubset['yhat'] = rf.predict(Xsubset)
    yhat = Xsubset['yhat'].values

    # print(Xsubset)
    same = 0
    for i in range(len(yhat)):
        if yhat[i] == Y[i]:
            same += 1

    print("ACCURACY: ", same/len(yhat))
