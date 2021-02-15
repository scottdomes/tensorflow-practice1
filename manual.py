import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.framework import tensor_shape

# This is me trying to manually build a simple neural network, step by step

# Cleaning and fetching data. See main.py for more info
url = './auto-mpg.data'
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
mpg = np.array(train_features['MPG'])

# STAGE ONE: Blind prediction
# The below code is the equivalent of...
# horsepower_model = tf.keras.Sequential([
#     layers.Dense(units=1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1))
# ])
# output = horsepower_model.predict(horsepower[:10])

# Use a set seed so I can compare my manual results consistently
# Not going to manually calculate the kernel, that's unnecessary
initializer = tf.keras.initializers.GlorotUniform(seed=1)
kernel = initializer(shape=(1, 1))
# tf.print(kernel)
# [[1.31112158]]
kernel_value = 1.31112158

# This data is the same as horsepower[:10]
horsepower_data = [[ 75.], [ 88.], [160.], [ 63.], [ 67.], [ 90.], [ 60.], [ 67.], [ 95.], [ 88.]]

results = []

for horsepower_unit in horsepower_data:
  # print(horsepower_unit, kernel_value)
  results.append([horsepower_unit[0] * kernel_value])

# print(results)
# [[98.3341185], [115.37869904], [209.7794528], [82.60065954], [87.84514586], [118.0009422], [78.66729480000001], [87.84514586], [124.5565501], [115.37869904]]


# STAGE TWO: Loss function + dummy optimizer + no plotting

# The below code is the equivalent of...
# horsepower_model.compile(
#     optimizer=DummyOptimizer(),
#     loss='mean_absolute_error')

# history = horsepower_model.fit(
#     train_features['Horsepower'][:10], train_labels[:10],
#     epochs=100, # Number of epochs really doesn't matter
#     # suppress logging
#     verbose=0)

# Which yields a mean absolute error of 26.834576

# See horsepower.py for construction of this code
horsepower_normalizer = preprocessing.Normalization()
horsepower_normalizer.adapt(horsepower)
kernel = initializer(shape=(1, 1))
input_value = horsepower[:10]
input_value = tf.reshape(input_value, [10, 1])
input_value = tf.cast(input_value, tf.float32)
normalized_value = horsepower_normalizer(horsepower[:10])
matrix_product = tf.tensordot(normalized_value, kernel, 1)
# tf.print(matrix_product)

horsepower_matrix = np.array(matrix_product)
# print(array)


def calculate_mae(data):
  mae_results = []

  for index, value in enumerate(data):
    horsepower_value = value
    mpg_value = mpg[index]
    mae_results.append(np.abs(mpg_value - horsepower_value))

  return np.mean(mae_results)

# print(calculate_mae(horsepower_matrix))
# 26.834576

# STAGE THREE: Loss function + dummy optimizer + plotting

def train_over_epochs(data, epochs = 100):
  epoch_results = []
  i = 0
  while i < epochs:
    epoch_results.append(calculate_mae(data))
    i = i + 1
  
  return epoch_results

def plot_loss(results):
  plt.plot(results, label='loss')
  plt.ylim([0, 30]) 
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)
  plt.show()

epoch_data = train_over_epochs(horsepower_matrix, 100)
print(epoch_data)
plot_loss(epoch_data)

# STAGE FOUR: Simple optimizer

# STAGE FIVE: Adam optimizer
