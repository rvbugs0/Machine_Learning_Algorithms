# loading the iris flower dataset and
from sklearn.datasets import load_iris
import numpy as np
import random
from MulticlassClassification import MulticlassClassification
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions

if __name__ == "__main__":

    iris = load_iris()
    X = iris.data
    Y = iris.target
    sepalLength = X[:, 0]
    sepalWidth = X[:, 1]
    petalLength = X[:, 2]
    petalWidth = X[:, 3]
    # print(iris)

    classes = [c for c in np.unique(Y)]
    n = int(0.1*X.shape[0]/len(classes))

    all_test_indices = []
    for c in classes:
        c_indices = np.where(Y == c)[0].tolist()
        c_indices = sorted(c_indices, key=(
            lambda x: x - random.randint(-100, 100)))
        all_test_indices = all_test_indices + c_indices[:n]

    test_X = X[all_test_indices]
    test_Y = Y[all_test_indices]

    mask = np.ones(X.shape[0], dtype=bool)
    mask[all_test_indices] = False
    train_X = X[mask]
    train_Y = Y[mask]

# data is now evenly split according to 1.1

    # sepal length / width
    model1 = MulticlassClassification(learning_rate=0.05, max_epochs=1000)
    model1.fit(train_X[:, :2], train_Y)
    print(model1.predict(test_X[:, :2]))
    print(accuracy_score(test_Y, model1.predict(test_X[:, :2])))

    # petal length / width
    model2 = MulticlassClassification(learning_rate=0.05, max_epochs=1000)
    model2.fit(train_X[:, -3:], train_Y)
    print(model2.predict(test_X[:, -3:]))
    print(accuracy_score(test_Y, model2.predict(test_X[:, -3:])))

    # all features
    model3 = MulticlassClassification(learning_rate=0.05, max_epochs=1000)
    model3.fit(train_X, train_Y)
    print(model3.predict(test_X))
    print(accuracy_score(test_Y, model3.predict(test_X)))

    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import itertools

    # Plotting Decision Regions
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=(10, 8))
    value = 1.5
    width = 0.75

    fig = plot_decision_regions(X=test_X[:, -3:],

                                filler_feature_values={
                                    2: value},
                                filler_feature_ranges={
                                    2: width},
                                y=test_Y, clf=model2, legend=2)

    # for clf, lab, grd in zip([model2, model2, model2, model2],
    #                         ['Logistic Regression', 'Random Forest', 'RBF kernel SVM', 'Ensemble'],
    #                         itertools.product([0, 1], repeat=2)):
    #     ax = plt.subplot(gs[grd[0], grd[1]])
    #     fig = plot_decision_regions(X=X[:,-3:], y=Y, clf=clf, legend=2)
    #     plt.title(lab)
    plt.show()
