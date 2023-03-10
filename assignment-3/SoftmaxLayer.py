import numpy as np
from Layer import Layer

class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        exp_input = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = exp_input / np.sum(exp_input, axis=1, keepdims=True)
        return self.output

    def backward(self, output_gradient):
        # Simplified backward pass for cross-entropy loss
        return output_gradient
