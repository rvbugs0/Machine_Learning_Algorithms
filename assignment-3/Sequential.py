from Layer import Layer


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
