import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import data set
house_data = pd.read_csv('house.csv')

# Convert to matrix
X = np.matrix([np.ones(house_data.shape[0]), house_data['surface'].as_matrix()]).T
y = np.matrix(house_data['loyer']).T

# Linear regression
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# Plot
plt.plot(house_data['surface'], house_data['loyer'], 'ro', markersize = 4)
plt.plot([0,250], [theta.item(0),theta.item(0)+250*theta.item(1)], linestyle='--')
plt.show()