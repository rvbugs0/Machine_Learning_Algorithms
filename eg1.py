import numpy as np

class LinearRegressionWithRegularization:
    def __init__(self, lr=0.01, num_iter=100000, reg_lambda=0.01, patience=5):
        self.lr = lr
        self.num_iter = num_iter
        self.reg_lambda = reg_lambda
        self.patience = patience
        
    def add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        X = self.add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size + self.reg_lambda * self.theta
            self.theta -= self.lr * gradient
            
            if i % 10000 == 0:
                z = np.dot(X, self.theta)
                h = self.sigmoid(z)
                print(f'loss: {self.loss(h, y)} \t')
                
    def predict_prob(self, X):
        X = self.add_intercept(X)
        
        return self.sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold=0.5):
        return self.predict_prob(X) >= threshold
