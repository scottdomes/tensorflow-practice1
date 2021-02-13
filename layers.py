import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


data = tf.constant([[[2,3], [1, 1]], [[2, 3], [1, 1]]])

# print(data)
# Normalize it
normalizer = preprocessing.Normalization()
normalizer.adapt(data)


# print(normalizer)
model = tf.keras.Sequential([
    layers.Dense(units=1, use_bias=False)
])

# model.summary()

# normalized = normalizer(data)
# print(normalized)

# print(model.bias)
# print(model.kernel)
# expected_output = activation(dot(input, kernel))
expected_output = tf.tensordot([[2,3], [1, 1]], [[2,3], [1, 1]], 2)
print(expected_output)

output = model.predict(data)

print(output)