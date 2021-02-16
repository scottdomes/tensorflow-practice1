import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
Y = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])

numerator = 0
denominator = 0
x_sum = np.sum(X)
y_sum = np.sum(Y)
for index, x_value in enumerate(X):
  y_value = Y[index]
  top_value = (x_value - x_sum) * (y_value - y_sum)
  numerator = numerator + top_value
  bottom_value = (x_value - x_sum) * (x_value - x_sum)
  denominator = denominator + bottom_value

slope = numerator / denominator

print(slope)
x = np.linspace(-4, 4, 200)
y = x**2

plt.xlabel('x')
plt.ylabel('y = x^2')
plt.plot(x, y)

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