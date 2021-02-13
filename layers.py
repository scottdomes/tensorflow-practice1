import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.framework import tensor_shape

input_value = tf.constant([[2.0,3.0], [1.0, 1.0]])

# print(np.matmul([2,3], [1, 1]))
# -> 5

dot_product = tf.tensordot([2.0,3.0], [1.0, 1.0], 1)
# print(dot_product)
# tf.Tensor(5.0, shape=(), dtype=float32)

initializer = tf.keras.initializers.GlorotUniform(seed=1)
kernel1 = initializer(shape=(2, 2))
# print(kernel1)
# tf.Tensor(
# [[ 0.9271029  -0.09802794]
#  [ 0.24566245 -0.2630809 ]], shape=(2, 2), dtype=float32)

kernel2 = initializer(shape=(2, 1))
# print(kernel2)
# tf.Tensor(
# [[ 1.0705262 ]
#  [-0.11319292]], shape=(2, 1), dtype=float32)

# With Kernel 1:
# matrix_product = tf.tensordot(input_value, kernel1, 1)
# print(matrix_product)
# tf.Tensor(
# [[ 2.5911932  -0.9852986 ]
#  [ 1.1727654  -0.36110884]], shape=(2, 2), dtype=float32)
# This is the same result as np.matmul(input_value, kernel),
# which generates a matrix product
# From the docs:
# Example 1: When a and b are matrices (order 2), the case axes = 1 is equivalent to matrix multiplication.
# Example 2: When a and b are matrices (order 2), the case axes = [[1], [0]] is equivalent to matrix multiplication.
# https://www.tensorflow.org/api_docs/python/tf/tensordot

# With Kernel 2:
matrix_product = tf.tensordot(input_value, kernel2, 1)
print(matrix_product)
# tf.Tensor(
# [[1.8014737]
#  [0.9573333]], shape=(2, 1), dtype=float32)


# DEPRECATED: Used to figure out what shape I wanted for the kernel
# input_shape = tensor_shape.TensorShape(matrix_product.shape)
# last_dim = tensor_shape.dimension_value(input_shape[-1])
# # print(input_shape)
# # (2, 2)
# # print(last_dim)
# # 1

model = tf.keras.Sequential([
    layers.Dense(units=1, use_bias=False, kernel_initializer=initializer, activation=None)
])
output = model.predict(input_value)
print(output)
# With units = 2
# [[ 2.5911932  -0.9852986 ]
#  [ 1.1727654  -0.36110884]]
# With units = 1
# [[1.8014737]
#  [0.9573333]]
