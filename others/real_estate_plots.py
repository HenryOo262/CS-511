import numpy as np
import pandas as p

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

df = p.read_csv(filepath_or_buffer='./real_estate_price_size_year.csv')

#print(df)
#print(df['price'].head(5))
#print(df['size'].describe())

y = df.iloc[:, :1] # output
#x = df.iloc[:, 1:] # input 2 features
x = df.loc[:, 'size']

sns.regplot(df, x='size', y='price')
plt.show()

# scikit learn's LR model expects 2d array cuz it is made for both LR and multiple reg

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=22)
print(x_train)
# x_train = x_train.values.reshape(-1, 1)
x_train = np.reshape(x_train, (-1, 1))
print(x_train)

model = LinearRegression()
model.fit(x_train, y_train)

x_test = np.reshape(x_test, (-1, 1))
p = model.predict(x_test)

'''

# 3d array

x = np.array(x)
x_test = np.array(x_test)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Assume 'x' has two features: 'size' and 'year'
ax.scatter(x[:, 0], x[:, 1], y, color='r')  # Plot the actual points

# Plot the predicted points (if you want to visualize them)
ax.plot_trisurf(x_test[:, 0], x_test[:, 1], p.flatten(), color='b', alpha=0.5)  # Use trisurf for surface

ax.set_xlabel('Size')
ax.set_ylabel('Year')
ax.set_zlabel('Price')

plt.show()
'''