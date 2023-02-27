import numpy as np

class LogisticRegression:

    def __init__(self, learning_rate=0.001, max_epochs=100):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.max_epochs):
            linear_model = X @ self.weights + self.bias
            hx = self._sigmoid(linear_model)

            dw = (X.T * (hx - y)).T.mean(axis=0)
            db = (hx - y).mean(axis=0)

            self.weights = self.weights - self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):

        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        return y_predicted

    def _sigmoid(self, x):
        return (1/(1+np.exp(-x)))
