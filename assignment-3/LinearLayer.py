import numpy as np
from Layer import Layer


class LinearLayer(Layer):
    def __init__(self, input_size, output_size):
        super().__init__(input_size,output_size)
        
    def forward(self, x):
        self.input = x
        output = np.dot(x, self.weights) + self.bias
        self.output = output
        return output

    def backward(self, error, learning_rate):
        grad_input = np.dot(error, self.weights.T)
        grad_weights = np.dot(self.input.T, error)
        grad_bias = np.sum(error, axis=0)
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        return grad_input
    



