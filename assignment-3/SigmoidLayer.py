from Layer import Layer
import numpy as np


class SigmoidLayer(Layer):
    def __init__(self, input_size):
        super().__init__(input_size, 1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_gradient(self, x):
        sigm = self.sigmoid(x)
        return sigm * (1 - sigm)

    def forward(self, x):
        self.input = x
        linear_output = np.dot(x, self.weights) + self.bias
        self.output = self.sigmoid(linear_output)
        return self.output

    def backward(self, error, learning_rate):
        grad_output = self.sigmoid_gradient(self.output) * error
        grad_weights = np.dot(self.input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0)
        grad_input = np.dot(grad_output, self.weights.T)
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        return grad_input
