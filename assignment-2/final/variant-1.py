from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from  MulticlassClassification import MulticlassClassification
from sklearn.metrics import accuracy_score
import numpy as np


if __name__ == '__main__':
    iris = load_iris()
    X, y = iris.data, iris.target
    print(y)

label_encoding = LabelEncoder()
y = label_encoding.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
clf = MulticlassClassification(learning_rate=0.001)

clf.fit(X_train, y_train)
print(clf.predict(X_test))
print(accuracy_score(y_test, clf.predict(X_test)))