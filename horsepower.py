import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.framework import tensor_shape

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

# Cleaning and fetching data. See main.py for more info
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)
dataset = raw_dataset.copy()
is_na = dataset.isna()
dataset = dataset.dropna()
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
train_features = train_dataset.copy()
horsepower = np.array(train_features['Horsepower'])

# Use a set seed so I can compare my manual results consistently
initializer = tf.keras.initializers.GlorotUniform(seed=1)

# UNNORMALIZED PREDICTIONS

# Model prediction
horsepower_model = tf.keras.Sequential([
    layers.Dense(units=1, kernel_initializer=initializer)
])

output = horsepower_model.predict(horsepower[:10])

print("UNNORMALIZED MODEL PREDICTION:")
print(output)

# Manual prediction
kernel = initializer(shape=(1, 1))
# Kernel shape should equal [last dimension of input shape, units]
input_value = horsepower[:10]
input_value = tf.reshape(input_value, [10, 1])
input_value = tf.cast(input_value, tf.float32)
matrix_product = tf.tensordot(input_value, kernel, 1)

print("UNNORMALIZED MANUAL PREDICTION:")
print(matrix_product)
# NORMALIZED PREDICTIONS

# Model prediction
horsepower_normalizer = preprocessing.Normalization()
horsepower_normalizer.adapt(horsepower)

horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1, kernel_initializer=initializer)
])

output = horsepower_model.predict(horsepower[:10])

print("NORMALIZED MODEL PREDICTION:")
print(output)

# Manual prediction
kernel = initializer(shape=(1, 1))
# Kernel shape should equal [last dimension of input shape, units]
input_value = horsepower[:10]
input_value = tf.reshape(input_value, [10, 1])
input_value = tf.cast(input_value, tf.float32)


normalized_value = horsepower_normalizer(horsepower[:10])
matrix_product = tf.tensordot(normalized_value, kernel, 1)

print("NORMALIZED MANUAL PREDICTION:")
tf.print(matrix_product)