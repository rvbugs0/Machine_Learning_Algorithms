# Importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# to compare our model's accuracy with sklearn model
# Logistic Regression


class LogitRegression():

    @staticmethod
    def sigmoid(x):
        # Activation function used to map any real value between 0 and 1
        return 1 / (1 + np.exp(-x))

    
    def net_input(self, x):
        # Computes the weighted sum of inputs Similar to Linear Regression
        return np.dot(x, self.weights)

    def probability(self, x):
        # Calculates the probability that an instance belongs to a particular class

        return self.sigmoid(self.net_input(x))

    def cost_function(self,x, y):
        # Computes the cost function for all the training samples
        m = x.shape[0]
        total_cost = -(1 / m) * np.sum(
            y * np.log(self.probability(x)) + (1 - y) * np.log(
                1 - self.probability(x)))
        return total_cost


    def __init__(self, learning_rate=0.01, max_epochs=100, batch_size=32, regularization=0,  patience=3,  bias=1):
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.bias = bias
        self.weights = None
        self.batch_validation_loss = np.array([[0, 0]])
        self.weights_array_wrt_epoch = []

    def fit(self, X, Y):
        # no_of_training_examples, no_of_features
        self.m, self.n = X.shape
        # weight initialization
        self.weights = np.zeros(self.n)
        self.bias = 0

        # setting aside 10% of data for validation set
        upperbound = int(self.m * 0.9)
        x = np.copy(X)[:upperbound]
        y = np.copy(Y)[:upperbound]

        validation_x = X[upperbound:]
        validation_y = Y[upperbound:]

        best_validation_loss = float('inf')
        wait = 0
        n, d = x.shape
        best_epoch = -1
        for epoch in range(self.max_epochs):

            print("Running epoch:", epoch+1, "/", self.max_epochs)

            # Split into batches
            for i in range(0, n, self.batch_size):
                X_batch = x[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]
                y_pred = np.dot(X_batch, self.weights)

                A = 1 / (1 + np.exp(- (X_batch.dot(self.weights) + self.bias)))

                # calculate gradients
                tmp = (A - y_batch.T)
                tmp = np.reshape(tmp, X_batch.shape[0])
                dW = np.dot(X_batch.T, tmp) / X_batch.shape[0]
                db = np.sum(tmp) / X_batch.shape[0]

                # update weights
                self.weights = self.weights - self.learning_rate * dW
                self.bias = self.bias - self.learning_rate * db
                


                batch_loss = self.cost_function(X_batch,y_batch)

                tup = [[self.batch_validation_loss.shape[0], batch_loss]]

                self.batch_validation_loss = np.append(
                    self.batch_validation_loss, tup, axis=0)

            self.weights_array_wrt_epoch.append(np.copy(self.weights))

            # Compute the validation loss
            
            validation_loss = self.cost_function(validation_x,validation_y)

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_epoch = epoch
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.batch_validation_loss = np.delete(
            self.batch_validation_loss, (0), axis=0)

        self.weights = self.weights_array_wrt_epoch[best_epoch]

        print("\nWeights", self.weights)
        print("Bias",self.bias)

    # Hypothetical function  h( x )

    def predict(self, X):
        Z = 1 / (1 + np.exp(- (X.dot(self.weights) + self.bias)))
        Y = np.where(Z > 0.5, 1, 0)
        return Y


# Driver code

def main():

    # Importing dataset
    df = pd.read_csv("diabetes.csv")
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1:].values

    # Splitting dataset into train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=1/3, random_state=0)

    # Model training
    model = LogitRegression(learning_rate=0.0001, patience=3, max_epochs=100)

    model.fit(X_train, Y_train)

    # Prediction on test set
    Y_pred = model.predict(X_test)

    # measure performance
    correctly_classified = 0

    # counter
    count = 0
    for count in range(np.size(Y_pred)):

        if Y_test[count] == Y_pred[count]:
            correctly_classified = correctly_classified + 1

        count = count + 1

    print("Accuracy on test set by our model       :  ", (
        correctly_classified / count) * 100)

    # print(model.batch_validation_loss)


if __name__ == "__main__":
    main()
