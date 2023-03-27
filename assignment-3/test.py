from NeuralNetwork import Sequential, LinearLayer, SigmoidLayer, SoftmaxLayer, CrossEntropyLoss, TanhLayer


model = Sequential()


model.add(LinearLayer(10,2))
model.add(LinearLayer(2,1))

print(model.layers[0].weights.shape)
print(model.layers[0].bias.shape)
a = model.layers[0].get_weights()
# print(model.layers[0].get_weights())
print(a)
# print(a[:-1, :])
# print(a[-1,:])

model.save_weights("best_weights")

model.load_weights("best_weights.npy")


print(model.layers[0].weights)