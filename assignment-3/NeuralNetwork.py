import numpy as np


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, X, y, learning_rate=0.1, epochs=1000):
        for epoch in range(epochs):
            output = self.predict(X)
            error = y - output

            for layer in reversed(self.layers):
                error = layer.backward(error, learning_rate)

    def save_weights(self, filename):
        weights = [layer.get_weights() for layer in self.layers]
        np.save(filename, weights)

    def load_weights(self, filename):
        weights = np.load(filename, allow_pickle=True)
        for i in range(len(self.layers)):
            self.layers[i].set_weights(weights[i])
