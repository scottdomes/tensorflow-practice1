import numpy as np
import matplotlib.pyplot as plt

# Say that y = x + 3
# But we don't know that, so all we know is y = x + theta
# The process:
# 1. Start with a random value of theta, say, 23
# 2. Calculate the mean squared error for all values of x using theta
# 3. Update theta to a new value. This is calculated via:
# theta = theta - learning_rate * the derivative of the mean squared error in respect to theta
# 4. Repeat until convergence

learning_rate = 0.1
x = np.linspace(1, 200, 200)
y = x + 3

def calculate_mse(theta):
  generator = [(x[i] + theta - y[i]) * (x[i] + theta - y[i]) for i in range(len(x))]
  array = np.asarray(generator)
  return 1/len(x) * sum(array)

def calculate_derivative(theta):
  generator = [x[i] + theta - y[i] for i in range(len(x))]
  array = np.asarray(generator)
  return 2/len(x) * sum(array)

possible_thetas = np.linspace(-5, 10, 200) 
possible_mses = np.asarray([calculate_mse(i) for i in possible_thetas]) 

x_axis = possible_thetas
y_axis = possible_mses

plt.xlabel('theta')
plt.ylabel('mse')
plt.plot(x_axis, y_axis)
plt.plot(x_axis, np.linspace(0,0, 200))

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
    plt.scatter(theta, mse)
    plt.pause(0.5)
    theta = theta - learning_rate * derivative

  plt.show()

gradient_descent(10, 20)