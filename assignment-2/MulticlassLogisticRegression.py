import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class MulticlassLogisticRegression:
    def __init__(self, lr=0.1, max_epoch=100, patience = 3,fit_intercept=True, verbose=False):
        self.lr = lr
        self.patience = patience
        self.max_epoch = max_epoch
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    def _add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        if self.fit_intercept:
            X = self._add_intercept(X)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        self.theta_ = np.zeros((X.shape[1], n_classes))

        for i in range(self.max_epoch):
            z = np.dot(X, self.theta_)
            h = self._softmax(z)
            gradient = np.dot(X.T, (h - (y == self.classes_).astype(int)))
            self.theta_ -= self.lr * gradient

            if self.verbose and i % 10000 == 0:
                z = np.dot(X, self.theta_)
                h = self._softmax(z)
                loss = self._cross_entropy_loss(h, y)
                print(f'Loss at iteration {i}: {loss}')

    def predict_proba(self, X):
        if self.fit_intercept:
            X = self._add_intercept(X)

        return self._softmax(np.dot(X, self.theta_))

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def _cross_entropy_loss(self, h, y):
        loss = -np.mean(np.sum(y * np.log(h), axis=1))
        return loss


if __name__ == '__main__':
    iris = load_iris()
    X, y = iris.data, iris.target

    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # train the model
    lr = MulticlassLogisticRegression(lr=0.1, max_epoch=50000, verbose=True)
    lr.fit(X_train, y_train)

    # predict the classes of test data
    y_pred = lr.predict(X_test)
    print(f'Test accuracy: {np.mean(y_pred == y_test)}')
