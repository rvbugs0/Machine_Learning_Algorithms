from sklearn.datasets import load_iris
import numpy as np
from LinearRegression import LinearRegression
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8.0, 6.0)
plt.xlabel("Step Number")
plt.ylabel("Batch Loss")
# plt.legend(labels=["Validation MSE","Entries trained"])


patience_for_all = 3
learning_rate_for_all = 0.005
regularization_for_all = 0.6


iris = load_iris()
iris_data = iris["data"]


# attributes - Sepal Length, Sepal Width, Petal Length, Petal Width
sepal_length = np.reshape(iris_data[:,0],(iris_data.shape[0],1))
sepal_width = np.reshape(iris_data[:,1],(iris_data.shape[0],1))
petal_length = np.reshape(iris_data[:,2],(iris_data.shape[0],1))
petal_width = np.reshape(iris_data[:,3],(iris_data.shape[0],1))

# Shuffle the data
n= iris_data.shape[0]
indices = np.arange(n)

# setting aside 10% of data for testing
upperbound = int(n * 0.9)

testing_set_indices = indices[upperbound:]
training_set_indices = indices[:upperbound]

indices  = indices[upperbound:]
np.random.shuffle(indices)


title = "Combination-1 : Predicting the petal width given petal length and sepal width"
print("\n\n")
print(title)
model1 = LinearRegression(learning_rate=0.0005,patience=patience_for_all,regularization=regularization_for_all)

sepal_width_training = sepal_width[training_set_indices]
petal_length_training = petal_length[training_set_indices]

X_input = np.concatenate((sepal_width_training,petal_length_training),axis=1)
Y_input = np.squeeze(np.copy(petal_width)[training_set_indices])


# fitting model-1
model1.fit(X_input,Y_input)

X_test_input = np.concatenate((sepal_width[testing_set_indices],petal_length[testing_set_indices]),axis=1)
Y_test_input = np.squeeze(np.copy(petal_width[testing_set_indices]))



model1.score(X_test_input,Y_test_input)
# print("\nTarget Values:",Y_test_input)
pred = model1.predict(X_test_input)
# print("\nPredictions:",pred)
for i  in range(Y_test_input.shape[0]):
    print("Target:",Y_test_input[i]," - Predicted:",pred[i])


x_plot = model1.batch_validation_loss[:,0]
y_plot = model1.batch_validation_loss[:,1]
plt.scatter(x_plot, y_plot)
plt.title(title)
plt.savefig('MODEL1_result.jpg')
plt.clf()
# plt.show()



print("\n\n###############################################################################################\n\n")


title = "Combination-2 : Predicting the petal width given petal length:"
print(title)

model2 = LinearRegression(learning_rate=learning_rate_for_all,patience=patience_for_all,regularization=regularization_for_all)

# sepal_width_training = np.copy(sepal_width)[training_set_indices]
petal_length_training = np.copy(petal_length)[training_set_indices]

X_input = petal_length_training
Y_input = np.squeeze(np.copy(petal_width)[training_set_indices])

# fitting model-2
model2.fit(X_input,Y_input)

X_test_input = np.copy(petal_length[testing_set_indices])
Y_test_input = np.squeeze(np.copy(petal_width[testing_set_indices]))

model2.score(X_test_input,Y_test_input)

pred = model2.predict(X_test_input)
# print("\nPredictions:",pred)
for i  in range(Y_test_input.shape[0]):
    print("Target:",Y_test_input[i]," - Predicted:",pred[i])

x_plot = model2.batch_validation_loss[:,0]
y_plot = model2.batch_validation_loss[:,1]
plt.scatter(x_plot, y_plot)
plt.title(title)
plt.savefig('MODEL2_result.jpg')

plt.clf()
# plt.show()


print("\n\n###############################################################################################\n\n")



title = "Combination-3 : Predicting the petal width given petal length, sepal length and sepal width:"
print(title)
model3 = LinearRegression(learning_rate=learning_rate_for_all,patience=patience_for_all,regularization=regularization_for_all)

sepal_width_training = np.copy(sepal_width)[training_set_indices]
petal_length_training = np.copy(petal_length)[training_set_indices]
sepal_length_training = np.copy(sepal_length)[training_set_indices]

X_input = np.concatenate((sepal_width_training,petal_length_training),axis=1)
X_input = np.concatenate((sepal_length_training,X_input),axis=1)
Y_input = np.squeeze(np.copy(petal_width)[training_set_indices])

# fitting model-3
model3.fit(X_input,Y_input)


sepal_width_testing = np.copy(sepal_width[testing_set_indices])
petal_length_testing = np.copy(petal_length)[testing_set_indices]
sepal_length_testing = np.copy(sepal_length)[testing_set_indices]


X_test_input = np.concatenate((sepal_width_testing,petal_length_testing),axis=1)
X_test_input = np.concatenate((sepal_length_testing,X_test_input),axis=1)


Y_test_input = np.squeeze(np.copy(petal_width[testing_set_indices]))

model3.score(X_test_input,Y_test_input)

pred = model3.predict(X_test_input)
# print("\nPredictions:",pred)
for i  in range(Y_test_input.shape[0]):
    print("Target:",Y_test_input[i]," - Predicted:",pred[i])

x_plot = model3.batch_validation_loss[:,0]
y_plot = model3.batch_validation_loss[:,1]
plt.scatter(x_plot, y_plot)
plt.title(title)
plt.savefig("MODEL3_result.jpg")
plt.clf()
# plt.show()


print("\n\n###############################################################################################\n\n")

title= "Combination-4: Predicting the sepal width given sepal length"
print(title)
model4 = LinearRegression(learning_rate=learning_rate_for_all,patience=patience_for_all,regularization=regularization_for_all)

sepal_length_training = sepal_length[training_set_indices]

X_input = sepal_length_training
Y_input = np.squeeze(np.copy(sepal_width)[training_set_indices])

# fitting model-4
model4.fit(X_input,Y_input)


sepal_length_testing = np.copy(sepal_length)[testing_set_indices]


X_test_input = sepal_length_testing
Y_test_input = np.squeeze(sepal_width[testing_set_indices])

model4.score(X_test_input,Y_test_input)

pred = model4.predict(X_test_input)
# print("\nPredictions:",pred)
for i  in range(Y_test_input.shape[0]):
    print("Target:",Y_test_input[i]," - Predicted:",pred[i])


x_plot = model4.batch_validation_loss[:,0]
y_plot = model4.batch_validation_loss[:,1]
plt.scatter(x_plot, y_plot)
plt.title(title)
plt.savefig("MODEL4_result.jpg")
plt.clf()

# plt.show()


print("\nPlots saved\nGoodbye!\n\n")