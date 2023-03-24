
import numpy as np
from NeuralNetwork import LinearLayer,Sequential,SigmoidLayer,CrossEntropyLoss

if __name__ == "__main__":
    # construct input data for XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # construct neural network
    model = Sequential()

    # layers = [2 (since we have 2 input features) , 8,1,4,1 ]

    model.add(LinearLayer(2, 32))
    model.add(SigmoidLayer())
    model.add(LinearLayer(32, 1))
    model.add(SigmoidLayer())


    model.train(X,y)

    X = np.array([[1,0],[0, 0], [0, 1], [1, 0], [1, 1],[0,0],[0,1]])
    # evaluate the trained model
    output = model.forward(X)

    print("Predictions:")
    binary_outputs = np.where(output >= 0.5, 1, 0)
    print(binary_outputs)


