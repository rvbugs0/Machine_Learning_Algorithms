import numpy as np

class Layer:
    def __init__(self,input_size=1, output_size=1):
        self.input = None
        self.output = None
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.bias = np.zeros(output_size)

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_gradient):
        raise NotImplementedError
    
    def get_weights(self):
        return np.array([self.weights,self.bias])
    
    def get_bias(self):
        return self.bias
    
    def set_weights(self,weights):
        self.weights = weights[0]
        self.bias = weights[1]

