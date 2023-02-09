import matplotlib.pyplot as plt
import seaborn as sns
from LinearRegression import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import scipy.io as sio

############################################## Loading data from matlab file ###################################################
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


train_X = pd.DataFrame()
for i in range(data.shape[0]):
    train_X=train_X.append(pd.DataFrame(data[i]))
locNames = ["Loc"+str(i) for i in range(1,37)]
train_X.insert(0,'LocName',locNames*data.shape[0])

train_Y = list(output_train[0])
for i in range(1,output_train.shape[0]):
    train_Y.extend(list(output_train[i]))
train_X.insert(0,'Target',train_Y)

new_train_X = pd.DataFrame()
for i in locNames:
    new_train_X=new_train_X.append(train_X[train_X['LocName']==i])
# print(new_train_X.head())
del train_X
del train_Y
train_X , train_Y = new_train_X.drop(['Target'],axis=1), new_train_X[['Target']]
# print(train_Y)
######
test_X = pd.DataFrame()
for i in range(test_data.shape[0]):
    test_X=test_X.append(pd.DataFrame(test_data[i]))
locNames = ["Loc"+str(i) for i in range(1,37)]
test_X.insert(0,'LocName',locNames*test_data.shape[0])

test_Y = list(output_test[0])
for i in range(1,output_test.shape[0]):
    test_Y.extend(list(output_test[i]))
test_X.insert(0,'Target',test_Y)

new_test_X = pd.DataFrame()
for i in locNames:
    new_test_X=new_test_X.append(test_X[test_X['LocName']==i])

del test_X
del test_Y
test_X , test_Y = new_test_X.drop(['Target'],axis=1), new_test_X[['Target']]
###############################################################################################################################
############################################ Model building ###################################################################
l = LinearRegression()
l.fit(train_X.drop(['LocName'],axis=1),train_Y['Target'],debug=True)
l.predict(test_X.drop(['LocName'],axis=1),debug=False)
test_mse = l.score(test_X.drop(['LocName'],axis=1),test_Y['Target'],debug=False)
print("\nScore methode MSE= ",test_mse)
sns.lineplot(data=[l.val_losses,l.train_losses]).set(xlabel='Step no.',ylabel='MSE',xlim=(0,None))
plt.legend(labels=["Validation MSE","Train MSE"])
plt.savefig('2nd_result.jpg')
plt.show()



