import numpy as np
# np.random.seed(23)

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
        if(self.bias.ndim==1):
            weights = np.append(self.weights,self.bias.reshape(1,self.bias.shape[0]),axis=0)
            return weights
        else:
            weights = np.append(self.weights,self.bias,axis =0)
            return weights

    def load_weights(self, w):
        self.weights = w[:-1, :]
        self.bias = w[-1,:]


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
        return x*(1-x)

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
        self.output = None

    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output

    def backward(self, y_true, learning_rate=0.001):
        grad = self.output - y_true
        grad /= y_true.shape[0]
        return grad


class TanhLayer(Layer):
    def __init__(self):
        super().__init__()

    def tanh(self, x):
        return np.tanh(x)
        # return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def forward(self, input):
        self.input = input
        output = self.tanh(input)
        return output

    def backward(self, output_gradient, learning_rate):
        tanh_grad = 1 - np.square(self.tanh(self.input))
        grad_input = output_gradient * tanh_grad
        return grad_input


class CrossEntropyLoss:
    def __init__(self):
        self.eps = 1e-8  # avoid taking log of zero

    def forward(self, y_pred, y_true):
        
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        self.y_pred = y_pred
        self.y_true = y_true
        loss = -np.mean(y_true * np.log(y_pred + self.eps) + (1 - y_true) * np.log(1 - y_pred + self.eps))
        return loss

    def backward(self):
        grad = -(self.y_true / self.y_pred) + (1 - self.y_true) / (1 - self.y_pred)
        grad /= self.y_true.shape[0]
        return grad






class Sequential():
    def __init__(self):
        super().__init__()
        self.layers = []
        self.epoch_training_validation_loss = np.vstack(np.array([[0,0,0]]))

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
        weights = [w.get_weights() for w in self.layers]
        weights = np.asanyarray(weights,dtype=object)
        np.save(filename,weights)        

    def load_weights(self, filename):
        weights = np.load(filename, allow_pickle=True)
        for i in range(len(self.layers)):
            self.layers[i].load_weights(weights[i])

    def predict(self, X):
        output =  self.forward(X)
        return output

    


    def train(self, X, y, learning_rate=0.05, epochs=10000000,patience = 3,loss_print_count=1,x_val=None,y_val=None):

        loss = CrossEntropyLoss()
        best_loss = 1000000000
        wait = 0
        best_weights = []
        
        # train neural network
        for i in range(epochs):
            output = self.forward(X)
            loss_val = loss.forward(output, y)

            weights = [layer.get_weights() for layer in self.layers]
            # backward pass
            grad = loss.backward()

            self.backward(grad, learning_rate)

            # check for convergence
            if loss_val < best_loss :
                best_weights = weights
                best_loss = loss_val
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

            if i % loss_print_count == 0:
                
                if(x_val is not None):

                    loss = CrossEntropyLoss()
                    val_output = self.forward(x_val)
                    val_loss = loss.forward(y_val, val_output)
                    tup = np.array([i,loss_val,val_loss])
                    # self.epoch_training_validation_loss = np.append(self.epoch_training_validation_loss,tup,axis=0)
                    self.epoch_training_validation_loss = np.vstack([self.epoch_training_validation_loss,tup])
                    print(f"Epoch {i}/{epochs}: loss = {loss_val:.4f}       Validation loss = {val_loss:.4f}",end="\r")
                    
                else:
                    print(f"Epoch {i}/{epochs}: loss = {loss_val}",end="\r")
        self.epoch_training_validation_loss = self.epoch_training_validation_loss[1:]
        print()

        for i in range(len(best_weights)):
            self.layers[i].load_weights(best_weights[i])


