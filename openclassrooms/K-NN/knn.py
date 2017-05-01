import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_mldata
from sklearn import neighbors

# Import MNIST dataset
mnist = fetch_mldata('MNIST original')

# Sampling (reduce the work)
sample = np.random.randint(70000, size=5000)
data = mnist.data[sample]
target = mnist.target[sample]

# Split training and testing
xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=0.8)

# Train the model
print('Training model')
errors = []
data = []
for k in range(2,15):
	# Import K-NN model
	knn = neighbors.KNeighborsClassifier(k)
	test = knn.fit(xtrain, ytrain)
	error = 100*(1-test.score(xtest, ytest))
	errors.append(error)
	data.append((error,k))

# Plot result & Pick the best K
k = min(data)[1]
print('Best k : {}'.format(k))
plt.ylabel('Error [%]')
plt.xlabel('K neighbors')
plt.plot(range(2,15),errors,'o-')
plt.show()

