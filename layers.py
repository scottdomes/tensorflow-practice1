import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

data = tf.constant([[2,3], [1, 1]])

print(np.matmul([2,3], [1, 1]))

model = tf.keras.Sequential([
    layers.Dense(units=0, use_bias=False, kernel_initializer=None, activation=None)
])

expected_output = tf.tensordot([2,3], [1, 1], 1)

print(expected_output)


initializer = tf.keras.initializers.GlorotUniform(seed=1)
values = initializer(shape=(2, 2))
print(values)
cast = tf.cast(values, tf.int32)
print(cast)
test = tf.tensordot(data, cast, [[2 - 1], [0]])
print(test)

output = model.predict(data)

print(output)