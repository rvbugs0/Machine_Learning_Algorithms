from math import log, exp
import numpy as np
import random
from DecisionTree import decision_tree_classifier


class AdaBoostClassifier:
    def __init__(self, n_estimators=10, learning_rate=0.01):

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.X = None
        self.Y = None
        self.classifiers = []
        self.weights = []

    def fit(self, X, Y):
        self.X = X.copy()
        self.Y = Y.copy()
        self.X['prob'] = 1/(self.X.shape[0])
        self.X['actual'] = self.Y
        self.X['misclassified'] = np.zeros(len(X))
        self.X['pred'] = np.zeros(len(X))

        for i in range(1, self.n_estimators+1):

            random.seed(i*10)

            classifier = decision_tree_classifier(max_depth=1)

            train_X = self.X.sample(
                len(self.X), replace=True, weights=self.X['prob'])

            
            classifier.fit(train_X.drop(['prob','actual','misclassified','pred'], axis=1), train_X['actual'])
            
            self.X['pred'] =  classifier.predict(self.X.drop(['prob','actual','misclassified','pred'], axis=1))
            
            self.X.loc[self.X.actual != self.X.pred, 'misclassified'] = 1
            self.X.loc[self.X.actual == self.X.pred, 'misclassified'] = 0

            error = sum(self.X['misclassified'] * self.X['prob'])
            
            alpha = self.learning_rate*0.5*log((1-error)/error)

            self.weights.append(alpha)
            self.classifiers.append(classifier)
            
            new_weight = self.X['prob'] * np.exp(-1*alpha*self.X['actual']*self.X['pred'])
            

            normalized_weight = new_weight/sum(new_weight)
            self.X['prob'] = round(normalized_weight, 4)
        
        return self
        

    def predict(self, X):
        t = np.zeros(X.shape[0])
        for classifier, weight in zip(self.classifiers, self.weights):
            
            predictions = np.array(classifier.predict(X))
            t += weight * predictions

        return np.sign(t)


if __name__ == '__main__':
    import pandas as pd
    # Reading data
    d = pd.read_csv(
        "decision-tree/data/train.csv")[['Age', 'Sex', 'Fare', 'Pclass', 'Survived']].dropna()
    d = d.assign(Sex=d.Sex.eq('male').astype(int))

    # Constructing the X and Y matrices
    X = d[['Age', 'Sex', 'Fare', 'Pclass']]
    Y = d['Survived'].values.tolist()

    ab = AdaBoostClassifier(10, 0.00001)
    ab.fit(X, Y)
    # Predicting

    Xsubset = X.copy()
    yhat = ab.predict(Xsubset)


    yhat = yhat.tolist()

    same = 0
    for i in range(len(yhat)):
        if yhat[i] == Y[i]:
            same += 1

    print(f"ACCURACY: {same/len(yhat):.5f}")

