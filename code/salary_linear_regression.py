import pandas as p
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = p.read_csv('./../data/SalaryData.csv')
print(df)

x = df.iloc[:, :1]
y = df.iloc[:, 1:]

sns.regplot(df, x='YearsExperience', y='Salary', color='b')
plt.show()

# split data 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(x_train, y_train)

# predict input is year
a = model.predict([[4]])

# predict with train data set
predict_trained = model.predict(x_train)
print(predict_trained)
# predict with test data set
predict_tested = model.predict(x_test)
print(predict_tested)

plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, predict_trained, color='blue')
plt.xlabel('yoe')
plt.ylabel('salary')
plt.box(False)
plt.show()

print('For trained set predict - ')
print(f'Slope {model.coef_}')
print(f'Intercept {model.intercept_}')

plt.scatter(x_test, y_test, color='red')
plt.plot(x_test, predict_tested, color='green')
plt.xlabel('yoe')
plt.ylabel('salary')
plt.box(False)
plt.show()

print('For tested set predict - ')
print(f'Slope {model.coef_}')
print(f'Intercept {model.intercept_}')