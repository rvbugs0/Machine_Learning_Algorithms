
# calculating derivative of loss wrt each layer's function

import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.eps = 1e-15  # avoid taking log of zero

    def forward(self, y_pred, y_true):
        m = y_true.shape[0]
        self.y_pred = y_pred
        self.y_true = y_true
        loss = -np.sum(y_true * np.log(y_pred + self.eps) +
                       (1 - y_true) * np.log(1 - y_pred + self.eps)) / m

        return loss

    def backward(self):
        m = self.y_true.shape[0]
        d_loss = (self.y_pred - self.y_true) / \
            (self.y_pred * (1 - self.y_pred) + self.eps) / m
        return d_loss


class Layer:
    def __init__(self, input_size=1, output_size=1):
        self.input = None
        self.output = None
        self.weights = np.random.randn(
            input_size, output_size) * np.sqrt(2 / input_size)
        self.bias = np.zeros((1, output_size))

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_gradient=None):
        raise NotImplementedError

    def get_weights(self):
        return np.array([self.weights, self.bias])

    def get_bias(self):
        return self.bias

    def set_weights(self, weights):
        self.weights = weights[0]
        self.bias = weights[1]


class LinearLayer(Layer):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)

    def forward(self, x):
        self.input = x
        self.output = x.dot(self.weights) + self.bias
        return self.output

    def backward(self, error, learning_rate):
        grad_input = np.dot(error, self.weights.T)

        grad_weights = np.dot(self.input.T, error)
        grad_bias = np.sum(error, axis=0, keepdims=True)
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias

        return grad_input


class SigmoidLayer(Layer):
    def __init__(self):
        super().__init__()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_gradient(self, x):
        sigm = self.sigmoid(x)
        return sigm * (1 - sigm)

    def forward(self, x):
        self.input = x
        self.output = self.sigmoid(x)
        return self.output

    def backward(self, error, learning_rate=0.001):
        grad_output = self.sigmoid_gradient(self.output) * error
        return grad_output


class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()
        self.softmax_output = None

    def forward(self, inputs):
        exp_inputs = np.exp(inputs)
        exp_inputs_sum = np.sum(exp_inputs, axis=1, keepdims=True)
        self.softmax_output = exp_inputs / exp_inputs_sum
        return self.softmax_output

    def backward(self, dout, learning_rate=0.001):
        # dout is the gradient of loss w.r.t. the output of the layer
        batch_size = self.softmax_output.shape[0]
        dscores = np.empty_like(self.softmax_output)

        for i in range(batch_size):
            jac = np.diagflat(
                self.softmax_output[i]) - np.outer(self.softmax_output[i], self.softmax_output[i])
            dscores[i] = np.dot(dout[i], jac)

        dscores /= batch_size
        # print("Dscore", dscores)
        return dscores


class TanhLayer(Layer):
    def __init__(self):
        super().__init__()

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, input):
        self.input = input
        output = self.tanh(input)
        return output

    def backward(self, output_gradient):
        tanh_grad = 1 - np.square(self.tanh(self.input))
        grad_input = output_gradient * tanh_grad
        return grad_input


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

    def backward(self, output_gradient, learning_rate):
        error = output_gradient
        for layer in reversed(self.layers):
            error = layer.backward(error, learning_rate)
        return error

    def save_weights(self, filename):
        weights = [layer.get_weights() for layer in self.layers]
        np.save(filename, weights)

    def load_weights(self, filename):
        weights = np.load(filename, allow_pickle=True)
        for i in range(len(self.layers)):
            self.layers[i].set_weights(weights[i])

    def predict(self, X):
        return model.forward(X)

    def train(self, X, y, learning_rate=0.1, epochs=1000):
        for epoch in range(epochs):
            output = self.predict(X)
            error = y - output

            for layer in reversed(self.layers):
                error = layer.backward(error, learning_rate)


if __name__ == "__main__":
    # construct input data for XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # construct neural network
    model = Sequential()

    # layers = [2 (since we have 2 input features) , 8,1,4,1 ]

    model.add(LinearLayer(2, 10))
    model.add(SigmoidLayer())
    model.add(LinearLayer(10, 2))
    model.add(LinearLayer(2, 1))
    model.add(SigmoidLayer())

    # model.add(SoftmaxLayer())
    loss = CrossEntropyLoss()

    # model.train(X, y)

    best_loss = 1000000000
    patience = 3
    wait = 0

    # train neural network
    learning_rate = 0.005
    for i in range(100000):
        # forward pass
        output = model.forward(X)
        loss_val = loss.forward(output, y)

        # backward pass
        grad = loss.backward()

        model.backward(grad, learning_rate)

        # # update weights
        # for layer in model.layers:
        #     if isinstance(layer, LinearLayer):
        #         layer.weights -= learning_rate * layer.grad_weights
        #         layer.biases -= learning_rate * layer.grad_biases

        # check for convergence
        if loss_val < best_loss:
            best_loss = loss_val
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

        if i % 1000 == 0:
            print(f"Iteration {i}: loss = {loss_val}")

    # evaluate the trained model
    output = model.forward(X)
    print("Predictions:")
    print(output)
