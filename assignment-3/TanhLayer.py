import numpy as np
from Layer import Layer
from LinearLayer import LinearLayer
from NeuralNetwork import NeuralNetwork

class TanhLayer(Layer):
    def __init__(self):
        super().__init__()
    
    def tanh(self, x):
        return np.tanh(x)
    
    def forward(self, input):
        self.input = input
        output = self.tanh(input)
        return output
    
    def backward(self, output_gradient, learning_rate):
        tanh_grad = 1 - np.square(self.tanh(self.input))
        grad_input = output_gradient * tanh_grad
        return grad_input

tanh_layer_1 = TanhLayer()
linear_layer_1 = LinearLayer(2, 2)
tanh_layer_2 = TanhLayer()
linear_layer_2 = LinearLayer(2, 1)

model_tanh = NeuralNetwork()
model_tanh.add_layer(linear_layer_1)
model_tanh.add_layer(tanh_layer_1)
model_tanh.add_layer(linear_layer_2
