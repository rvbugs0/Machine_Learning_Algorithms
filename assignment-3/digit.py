from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

epochs = 1000
learning_rate =  0.5

# Preprocess the data
X_train = X_train.reshape((60000, 784)) / 255.0
X_test = X_test.reshape((10000, 784)) / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Create a validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

print(X_train.shape)

from NeuralNetwork import Sequential, LinearLayer, SigmoidLayer, SoftmaxLayer, CrossEntropyLoss, TanhLayer

# Create the neural network
model = Sequential()

# Add layers to the neural network
model.add(LinearLayer(input_size=784, output_size=64))
model.add(TanhLayer())
model.add(LinearLayer(input_size=64, output_size=16))
model.add(SigmoidLayer())
model.add(LinearLayer(16,10))
model.add(SoftmaxLayer())

# Define the loss function
loss = CrossEntropyLoss()

# Train the neural network
for epoch in range(epochs):
    # Forward pass
    y_pred = model.forward(X_train)
    train_loss = loss.forward(y_pred, y_train)

    # Backward pass
    error = loss.backward()
    model.backward(error, learning_rate)

    # Compute validation loss
    if epoch % 5 == 0:
        y_val_pred = model.predict(X_val)
        val_loss = loss.forward(y_val_pred, y_val)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

# Evaluate the neural network on the test set
y_test_pred = model.predict(X_test)
test_loss = loss.forward(y_test_pred, y_test)
print(f"Test Loss={test_loss:.4f}")

best_val_loss = float("inf")
wait = 0

for epoch in range(epochs):
    # Forward pass
    y_pred = model.forward(X_train)
    train_loss = loss.forward(y_pred, y_train)

    # Backward pass
    error = loss.backward()
    model.backward(error, learning_rate)

    # Compute validation loss
    y_val_pred = model.predict(X_val)
    val_loss = loss.forward(y_val_pred, y_val)

    # Check if validation loss has improved
