import numpy as np
from LogisticRegression import LogisticRegression
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class MulticlassClassification:

    def __init__(self, learning_rate=0.005, patience=3, max_epochs=1000):
        self.models = []
        self.learning_rate = learning_rate
        self.patience = patience
        self.max_epochs = max_epochs


    def fit(self, X, y):
        """
        Fits each model
        """
        for y_i in np.unique(y):
            # y_i - positive class for now

            # All other classes except y_i are negative

            # Choose x where y is positive class
            x_true = X[y == y_i]
            # Choose x where y is negative class
            x_false = X[y != y_i]
            # Concatanate
            x_true_false = np.vstack((x_true, x_false))

            # Set y to 1 where it is positive class
            y_true = np.ones(x_true.shape[0])
            # Set y to 0 where it is negative class
            y_false = np.zeros(x_false.shape[0])
            # Concatanate
            y_true_false = np.hstack((y_true, y_false))

            # Fit model and append to models list
            model = LogisticRegression(learning_rate=self.learning_rate,max_epochs=self.max_epochs )
            model.fit(x_true_false, y_true_false)
            self.models.append([y_i, model])

    def predict(self, X):
        y_pred = [[label, model.predict(X)] for label, model in self.models]

        output = []

        for i in range(X.shape[0]):
            max_label = None
            max_prob = -10**5
            for j in range(len(y_pred)):
                prob = y_pred[j][1][i]
                if prob > max_prob:
                    max_label = y_pred[j][0]
                    max_prob = prob
            output.append(max_label)

        return output



if __name__ == "__main__":
    X, y = make_blobs(n_samples=20, n_features=2, centers=3, random_state=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    plt.scatter(X[:, 0], X[:, 1], c=np.array(['r', 'g', 'b'])[y])
    plt.show()

    model = MulticlassClassification(max_epochs=1000)

    model.fit(X_train, y_train)
    print(model.predict(X_test))
    print(accuracy_score(y_test, model.predict(X_test)))