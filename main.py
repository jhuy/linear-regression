import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Initialize parameters of code
lowest_int = 1
highest_int = 5
num_rand_ints = 100
noise_factor = 1.5
num_interations_of_regression = 50

# intercept
theta0 = 0
# slope
theta1 = 0
# learning rate
alpha = 0.1

# Generates random numbers to be linear regressed
x_coor = np.linspace(lowest_int,highest_int, num = num_rand_ints)
y_coor = np.linspace(lowest_int,highest_int*1.25, num = num_rand_ints)

def noise(k):
    return k+((np.random.random()*2)-1)*noise_factor

y_coor = np.vectorize(noise)(y_coor)

data = []
for index in range (0, num_rand_ints):
    data.append([x_coor[index], y_coor[index]])

df = pd.DataFrame (data, columns=['x','y'])
df.plot(kind='scatter', x='x', y='y')

# Generates the line of best fit, will be filled in later
line_of_best_fit = list(range(lowest_int, highest_int))
plt.plot(line_of_best_fit)
plt.show()

def new_theta_function (alpha, theta0, theta1, x, y):
    new_theta0 = float(theta0 - (alpha/num_rand_ints) * sum_theta_function_x(theta0, theta1, x, y))
    new_theta1 = float(theta1 - (alpha/num_rand_ints) * sum_theta_function_y(theta0, theta1, x, y))
    print ([new_theta0, new_theta1])
    return [new_theta0, new_theta1]

#sum(theta0 + theta1*x - y)
def sum_theta_function_x (theta0, theta1, x, y):
    total = 0
    for index in range(0, num_rand_ints):
        total += float(theta0 + theta1*x.iloc[index] - y.iloc[index])
    return total

#sum((theta0 + theta1*x -y)*x)
def sum_theta_function_y (theta0, theta1, x, y):
    total = 0
    for index in range(0, num_rand_ints):
        total += float((theta0 + theta1*x.iloc[index] - y.iloc[index])*x.iloc[index])
    return total

for index in range(num_interations_of_regression):
    [theta0, theta1] = new_theta_function(alpha, theta0, theta1, df.loc[:, 'x'], df.loc[:, 'y'])