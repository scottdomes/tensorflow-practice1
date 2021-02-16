import numpy as np
import matplotlib.pyplot as plt

# Say that y = x + 3
# But we don't know that, so all we know is y = x + theta
# The process:
# 1. Start with a random value of theta, say, 23
# 2. Calculate the mean squared error for all values of x using theta
# 3. Update theta to a new value. This is calculated via:
# theta = theta - learning_rate * the derivative of the mean squared error
# 4. Repeat until convergence

# The derivative of the mean squared error is in respect to theta

theta = 4
learning_rate = 0.1
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

def calculate_mse(theta):
  generator = [(x[i] + theta - y[i]) * (x[i] + theta - y[i]) for i in range(len(x))]
  array = np.asarray(generator)
  return 1/len(x) * sum(array)

def calculate_derivative(theta):
  generator = [x[i] + theta - y[i] for i in range(len(x))]
  array = np.asarray(generator)
  return 2/len(x) * sum(array)

# mse = calculate_mse()
# derivative = calculate_derivative()
# theta = theta - learning_rate * derivative
# print(theta)
# print(mse)


def gradient_descent(initial_theta, epochs):
  theta_values = []
  mse_values = []
  theta = initial_theta
  for __ in range(epochs):
    mse = calculate_mse(theta)
    mse_values.append(mse)
    theta_values.append(theta)
    print(mse)
    print(theta)
    print('----')
    derivative = calculate_derivative(theta)
    theta = theta - learning_rate * derivative

gradient_descent(4, 20)

# t0 = 0
# t1 = 0
# m = 10
# grad0 = 1.0/m * sum([(t0 + t1*np.asarray([x[i]]) - y[i]) for i in range(m)])
# grad1 = 1.0/m * sum([(t0 + t1*np.asarray([x[i]]) - y[i])*np.asarray([x[i]]) for i in range(m)])

# print(grad0)
# print(grad1)
# x = np.linspace(-4, 4, 200)
# y = x**2

# plt.xlabel('x')
# plt.ylabel('y = x^2')
# plt.plot(x, y)

# learning rate
lr = 0.1
np.random.seed(20)
x_start = np.random.normal(0, 2, 1)
dy_dx_old = 2 * x_start
dy_dx_new = 0

tolerance = 1e-2
# stop once the value has converged
# while abs(dy_dx_new - dy_dx_old) > tolerance:
#     dy_dx_old = dy_dx_new
#     x_start = x_start - lr * dy_dx_old
#     dy_dx_new = 2 * x_start
#     plt.scatter(x_start, x_start**2)
#     plt.pause(0.5)
# plt.show()