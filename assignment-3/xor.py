import numpy as np
from Sequential import Sequential
from LinearLayer import LinearLayer
from SigmoidLayer import SigmoidLayer
from TanhLayer import TanhLayer
from CrossEntropyLoss import CrossEntropyLoss

# construct input data for XOR problem
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# construct neural network
model = Sequential()
model.add(LinearLayer(2, 2))
model.add(SigmoidLayer(2))
model.add(LinearLayer(2, 1))
model.add(SigmoidLayer(2))
loss = CrossEntropyLoss()

# train neural network
learning_rate = 0.1
for i in range(10000):
    # forward pass
    output = model.forward(X)
    loss_val = loss.forward(output, y)

    # backward pass
    grad = loss.backward()
    model.backward(grad)

    # update weights
    for layer in model.layers:
        if isinstance(layer, LinearLayer):
            layer.weights -= learning_rate * layer.grad_weights
            layer.biases -= learning_rate * layer.grad_biases

    # check for convergence
    if i % 1000 == 0:
        print(f"Iteration {i}: loss = {loss_val}")

# evaluate the trained model
output = model.forward(X)
print("Predictions:")
print(output)
