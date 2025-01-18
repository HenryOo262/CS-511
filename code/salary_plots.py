import pandas as p
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = p.read_csv('./../data/SalaryData.csv')
print(df.head(n=5)) # returns first 5 rows
print(df.describe()) # aggregates

salary = df['Salary']
yoe = df['YearsExperience']
salary = np.array(salary) # turn to array (optional)
yoe = np.array(yoe)
print(salary)
print(yoe)

sns.histplot(df['Salary']) # histogram of salary
plt.show()

# Distribution
plt.title('Salary Disribution Plot')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.plot(df['YearsExperience'], df['Salary'])
# plt.plot(yoe, salary)
plt.show()

# RS between x and y data
plt.scatter(df['YearsExperience'], df['Salary'], color='red')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.box(False) # disable box border
plt.show()

# independent data/ input
# [::] - start, stop, step
x = df.iloc[:, :1] # all rows, column 0 only (years)
y = df.iloc[:, 1:] # all rows, only column 1 (salary)

