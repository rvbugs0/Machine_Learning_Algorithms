import numpy as np
import pandas as pd
import scipy.io as sio
from LinearRegression import LinearRegression
import math
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8.0, 6.0)
plt.title("Predicting traffic flow for 36 spatial locations 15 minutes into the future")
plt.xlabel("Step Number")
plt.ylabel("Batch Loss")

import warnings
warnings.filterwarnings("ignore")


data_path = "data/traffic_dataset.mat"

# The provided data is given in a `.mat` file.
# We can load it using the `scipy.io.loadmat` function.
mat = sio.loadmat(data_path)
input_train = mat['tra_X_tr']
output_train = mat['tra_Y_tr'].T
input_test = mat['tra_X_te']
output_test = mat['tra_Y_te'].T
adj_mat = mat['tra_adj_mat']

input_train = input_train.squeeze()
input_test = input_test.squeeze()

# Convert the sparse matrix to a dense matrix
data = []
test_data = []

for i in range(input_train.shape[0]):
    data.append(input_train[i].todense())

for i in range(input_test.shape[0]):
    test_data.append(input_test[i].todense())

# Convert the data to a numpy array
data = np.array(data)
test_data = np.array(test_data)


print("---------------- Prepairing training set -----------------")

training_set_x = pd.DataFrame()

# total 1261 entries of data for every location -> total = 36 * 1261 = 45396
# each entry is a 36x48 table
total_entries = data.shape[0]

for i in range(total_entries):
    training_set_x = training_set_x.append(pd.DataFrame(data[i]))


train_y_col = []

for i in range(output_train.shape[0]):
    train_y_col.extend(list(output_train[i]))

training_set_x.insert(0,'Target',train_y_col)

tr_x = pd.DataFrame()
tr_y = pd.DataFrame()

for i in range(total_entries):
    for j in range(36):
        n = i + j*36        
        tr_x = tr_x.append(training_set_x.iloc[n])
        

del training_set_x
del train_y_col


tr_y = tr_x.iloc[:,0]
tr_x.drop(tr_x.columns[0],axis=1,inplace=True)

print("---------------- Prepairing testing set -----------------")


testing_set_x = pd.DataFrame()

for i in range(test_data.shape[0]):
    testing_set_x = testing_set_x.append(pd.DataFrame(test_data[i]))

test_y_col = []

for i in range(output_test.shape[0]):
    test_y_col.extend(list(output_test[i]))

testing_set_x.insert(0,'Target',test_y_col)

te_x = pd.DataFrame()
te_y = pd.DataFrame()

for i in range(test_data.shape[0]):
    for j in range(36):
        n = i + j*36        
        te_x = te_x.append(testing_set_x.iloc[n])
        

del testing_set_x
del test_y_col

te_y = te_x.iloc[:,0]
te_x.drop(te_x.columns[0],axis=1,inplace=True)


print("---------------- Initiating model -----------------")

l = LinearRegression(learning_rate=0.005,regularization=0.01)
l.fit(tr_x,tr_y)
l.predict(te_x)
test_mse = l.score(te_x,te_y)
print("\nScore method RMSE= ",math.sqrt(test_mse))


x_plot = l.batch_validation_loss[:,0]
y_plot = l.batch_validation_loss[:,1]
plt.scatter(x_plot, y_plot)
plt.savefig("TrafficData.png")
# plt.show()
plt.clf()
print("\nPlots saved\nGoodbye!\n\n")


