from tensorflow.keras.layers.experimental import preprocessing
import pandas as pd
import numpy as np
import tensorflow as tf
from dummy_optimizer import DummyOptimizer
from tensorflow.keras import layers

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
train_labels = train_features.pop('MPG')
horsepower = np.array(train_features['Horsepower'])
mpg = np.array(test_dataset['MPG'])

# Create normalizer
horsepower_normalizer = preprocessing.Normalization()
horsepower_normalizer.adapt(horsepower)
initializer = tf.keras.initializers.GlorotUniform(seed=1)

# STAGE TWO: Loss function + dummy optimizer + no plotting

# The below code is the equivalent of...
# horsepower_model = tf.keras.Sequential([
#     horsepower_normalizer,
#     layers.Dense(units=1, kernel_initializer=initializer)
# ])
# horsepower_model.compile(
#     optimizer=DummyOptimizer(),
#     loss='mean_absolute_error')
# history = horsepower_model.fit(
#     train_features['Horsepower'][:10], train_labels[:10],
#     epochs=10, # Number of epochs really doesn't matter
#     # suppress logging
#     verbose=0)
# print(history.history)
# Which yields a mean absolute error of 26.834576

# See horsepower.py for construction of this code
kernel = initializer(shape=(1, 1))
normalized_value = horsepower_normalizer(horsepower[:10])
matrix_product = tf.tensordot(normalized_value, kernel, 1)

horsepower_matrix = np.array(matrix_product)
mpg_array = np.asarray(train_labels[:10])

def calculate_mae(data):
  mae_results = []

  for index, horsepower_value in enumerate(data):
    mpg_value = mpg_array[index]
    print(f"{mpg_value} {horsepower_value}")
    mae_results.append(np.abs(mpg_value - horsepower_value))	

  return np.mean(mae_results)

print(calculate_mae(horsepower_matrix))
# 26.834576