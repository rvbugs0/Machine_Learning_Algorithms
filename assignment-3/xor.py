
import numpy as np
from NeuralNetwork import LinearLayer,Sequential,SigmoidLayer,TanhLayer

if __name__ == "__main__":
    # construct input data for XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    while(True):
        choice = input("Enter:\n1 for Tanh:\n2 for Sigmoid (can take multiple attempts to reduce loss):\n3 to exit\n")
        choice = int(choice)
        # construct neural network
        model = Sequential()

        if(choice==1):


        # layers = [2 (since we have 2 input features) , 8,1,4,1 ]
    
            # with tanh layer
            model.add(LinearLayer(2, 2))
            model.add(TanhLayer())
            model.add(LinearLayer(2, 16))
            model.add(TanhLayer())
            model.add(LinearLayer(16, 1))
            model.add(SigmoidLayer())
            model.train(X,y,learning_rate=0.05,patience=5,epochs=10000)

        if(choice==2):
            # with all sigmoid layers
            model.add(LinearLayer(2, 2))
            model.add(SigmoidLayer())
            model.add(LinearLayer(2, 16))
            model.add(SigmoidLayer())
            model.add(LinearLayer(16, 1))
            model.add(SigmoidLayer())
            model.train(X,y,learning_rate=0.1,patience=5,epochs=10000)
        
        if(choice==3):
            break


        

        # evaluate the trained model
        output = model.forward(X)

        print("Predictions:")
        binary_outputs = np.where(output >= 0.5, 1, 0)
        print(binary_outputs)


    # also need to save weights
