from Layer import Layer
import numpy as np

class Sequential(Layer):
    def __init__(self):
        super().__init__()
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        self.output = output
        return output

    def backward(self, output_gradient):
        error = output_gradient
        for layer in reversed(self.layers):
            error = layer.backward(error)
        return error
    
    def save_weights(self, filename):
        weights = [layer.get_weights() for layer in self.layers]
        np.save(filename, weights)

    def load_weights(self, filename):
        weights = np.load(filename, allow_pickle=True)
        for i in range(len(self.layers)):
            self.layers[i].set_weights(weights[i])

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

