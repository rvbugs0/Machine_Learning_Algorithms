# Making the imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


from LinearRegression import LinearRegression as LR


plt.rcParams['figure.figsize'] = (12.0, 9.0)
X = np.array([i for i in range(1,101)])
X = np.reshape(X,[100,1])

print(X)

upperbound = int(X.size * 0.8)
print(X[:upperbound])



# Y = np.array([i*0.1*(random.randint(-1,1))*(random.randint(-5,5)) +2   for i in range(1,101)])
# # print(Y)
# X = np.reshape(Y,[100,1])
# plt.scatter(X, Y)
# plt.show()

# lr = LR()
# lr.fit(X,Y)
# lr.predict(X)
# lr.score(X,Y)









# import numpy as np

# a = np.array([[1,2,3],[4,5,6],[7,8,9]])
# n, d = a.shape
# b = np.zeros(d)

# indices = np.arange(n)
# np.random.shuffle(indices)
# print(indices)
# X = a[indices]
# y = a[indices]

# print(X)
