from matplotlib import pyplot as plt
import pandas as p
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = p.read_csv('./sat.csv')
print(df)
# print(df.describe())

# students who perform well on SAT tends to have better gpa - EXPERT KNOWLEDGE FROM CHATGPT LMAO

# sns.scatterplot(data=df, x='SAT', y='GPA')
sns.regplot(data=df, x='SAT', y='GPA')
plt.show()

plt.scatter(df['SAT'], df['GPA'])
plt.title('academic data')
plt.xlabel('SAT score')
plt.ylabel('GPA')
plt.show()

# input
x = df.iloc[:,:1]
# label
y = df.iloc[:,1:]

print(len(x))
print(len(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print(len(x_train))
print(len(y_train))

model = LinearRegression()
model.fit(X=x_train, y=y_train)

predict = model.predict(x_test)
print(predict)

plt.scatter(x_test, y_test, color='blue')
plt.plot(x_test, predict, color='red')
plt.xlabel('SAT')
plt.ylabel('GPA')
plt.show()