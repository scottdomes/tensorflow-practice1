import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from dummy_optimizer import DummyOptimizer

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


# url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
url = './auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

# Copy to avoid modifying initial data
# https://stackoverflow.com/questions/27673231/why-should-i-make-a-copy-of-a-data-frame-in-pandas
dataset = raw_dataset.copy()

# Print last five rows
dataset.tail()

# Returns a version of the dataset where all the values are True or False, 
# depending if null (True meaning is null)
is_na = dataset.isna()

# Sums the columns. Since our dataset is now true/false,
# true values are 1, false are 0
# Note: does not modify dataset
is_na.sum()

# Drops all null rows
dataset = dataset.dropna()

# print(dataset)

# Maps country codes (1, 2 ,3) to actual names
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})

# print(dataset)

# A dummy variable in statistics is a variable that is either 1 or 0, AND
# indicates some important aspect of the data.
# It can be used to sort rows into mutually exclusive categories.
# Here, we convert a single Origin column with values of 1, 2, or 3
# into a single Origin column with values 'USA', 'Europe', 'Japan'
# and THEN into three columns (USA, Europe, Japan) with 1 or 0 values.
# TODO: figure out why we did this
dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')

# Splits data into train and test sets. 
# https://developers.google.com/machine-learning/crash-course/training-and-test-sets/splitting-data#:~:text=training%20set%E2%80%94a%20subset%20to,to%20test%20the%20trained%20model.
# One larger piece (80%) will be used to train the model, and the smaller set will be used to test it
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Plots the data into a pairwise plot, which shows the variables
# in relation to each other. Good for establishing relationships
# sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')

# plt.show()

# Shows mean, count, standard deviation for each column
train_dataset.describe().transpose()

# Show just mean and std
train_dataset.describe().transpose()[['mean', 'std']]

# We can also represent our values as a tensor.
# This produces a 2D tensor of shape 314, 9. In English that means
# it creates an array of arrays. There are 314 arrays, and each is 9 items long
# A 2D tensor is AKA a matrix, btw
# If we limit it to just the first five rows...

# print(np.array(train_dataset[:5]))

# ... it looks like so:
# [[   4.    90.    75.  2125.    14.5   74.     0.     0.     1. ]
#  [   4.   140.    88.  2890.    17.3   79.     0.     0.     1. ]
#  [   8.   350.   160.  4456.    13.5   72.     0.     0.     1. ]
#  [   4.   105.    63.  2125.    14.7   82.     0.     0.     1. ]
#  [   4.    97.    67.  2145.    18.    80.     0.     1.     0. ]]
# Note that this tensor is of shape 5, 9, since we only took the first 5 values in our dataset.

# We want to predict MPG, so we need to split it away from the other features
train_features = train_dataset.copy()
test_features = test_dataset.copy()
train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

# Our data has lots of variance in range, e.g. from train_features:
#                      mean         std
# Cylinders        5.477707    1.699788
# Displacement   195.318471  104.331589
# Horsepower     104.869427   38.096214
# Weight        2990.251592  843.898596
# (this comes from print(train_features.describe().transpose()[['mean', 'std']]))
# To make our training more stable, we should normalize it
# Our goal is to have a mean of 0 and a variance of 1 for every value
# To do that, we make certain values more important, and others less
# So in this case, we'd tell our model to de-emphasize displacement
# and overemphasize cylinders, and thus they'd be given equal importance
# note: std is the square root of variance,
# variance is the average degree to which each point differs from the mean

# Create the normalizer
normalizer = preprocessing.Normalization()
# Teaches the normalizer to coerce the inputs to mean 0 variance 1
normalizer.adapt(np.array(train_features))

# Produces a tensor of normalized values from our dataset
tensor = normalizer(train_features)

# Remember our original tensor above? It looked like this:
# [[   4.    90.    75.  2125.    14.5   74.     0.     0.     1. ]
#  [   4.   140.    88.  2890.    17.3   79.     0.     0.     1. ]
#  [   8.   350.   160.  4456.    13.5   72.     0.     0.     1. ]
#  [   4.   105.    63.  2125.    14.7   82.     0.     0.     1. ]
#  [   4.    97.    67.  2145.    18.    80.     0.     1.     0. ]]

# Well, check out the normalized version...

# print(tensor[:5])

# [[-0.871 -1.011 -0.785 -1.027 -0.38  -0.517 -0.466 -0.496  0.776]
#  [-0.871 -0.531 -0.444 -0.119  0.625  0.845 -0.466 -0.496  0.776]
#  [ 1.486  1.485  1.449  1.74  -0.739 -1.062 -0.466 -0.496  0.776]
#  [-0.871 -0.867 -1.101 -1.027 -0.309  1.663 -0.466 -0.496  0.776]
#  [-0.871 -0.944 -0.996 -1.003  0.876  1.118 -0.466  2.016 -1.289]]

# Look at the first column. 4 cylinders becomes -0.871, 8 cylinders becomes 1.486.
# In column 2, 90 displacement becomes -1.011, and 350 becomes 1.485.

# Recall our old std and mean?
#                      mean         std
# Cylinders        5.477707    1.699788
# Displacement   195.318471  104.331589
# Horsepower     104.869427   38.096214
# Weight        2990.251592  843.898596
# What's the new standard deviation? Well, let's see
# These commands produce the STD and mean for each column

# print(tf.math.reduce_mean(tensor, 0))
# print(tf.math.reduce_std(tensor, 0))

# The 0 indicates we're doing this calculation across the first axis
# An example is helpful here. Pretend our tensor is [[1, 1], [2, 2]]
# tf.math.reduce_mean(tensor, 0) would produce [1.5, 1.5]. That's the average for each column
# tf.math.reduce_mean(tensor, 1) would produce [1, 2]. That's the average for each item
# So if we ran tf.math.reduce_mean(tensor, 1) with our MPG data, we'd get an array 314 items long,
# and the average would for cylinders + displacements + horsepower, etc... not very useful
# Anyway, here's what these two commands give us:
# tf.Tensor([ 0.  0. -0. -0.  0.  0. -0. -0.  0.], shape=(9,), dtype=float32)
# tf.Tensor([1. 1. 1. 1. 1. 1. 1. 1. 1.], shape=(9,), dtype=float32)
# The mean is 0 for each column, and the variance is 1. Exactly what we wanted.

# Now that we know how to normalize our dataset, we want to train our model to
# predict MPG from horsepower. 

# Get the horsepower
horsepower = np.array(train_features['Horsepower'])

# Normalize it
horsepower_normalizer = preprocessing.Normalization()
horsepower_normalizer.adapt(horsepower)

# For a detail breakdown of this, see horsepower.py
# For a manual implementation of the below lines, see manual.py
horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1)) # TODO: Remove initialzier
])

horsepower_model.summary()

# Shows you the expected shape of the outcome: (10, 1)
# Values may differ based on initializer used for the dense layer
# See horsepower.py for how to use a fixed initializer
predict = horsepower_model.predict(horsepower[:10])
# tf.print(predict)
# array([[ 1.202],
#        [ 0.679],
#        [-2.218],
#        [ 1.685],
#        [ 1.524],
#        [ 0.598],
#        [ 1.805],
#        [ 1.524],
#        [ 0.397],
#        [ 0.679]], dtype=float32)

x = tf.linspace(0.0, 10, 11) # Should be 250, 251 for 2nd and 3rd args
y = horsepower_model.predict(x)
# tf.print(y)
# This result is consistent
# array([[-3.615],
#        [-3.58 ],
#        [-3.546],
#        [-3.512],
#        [-3.477],
#        [-3.443],
#        [-3.408],
#        [-3.374],
#        [-3.339],
#        [-3.305],
#        [-3.27 ]], dtype=float32)


def plot_horsepower(x, y):
  plt.scatter(train_features['Horsepower'], train_labels, label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('Horsepower')
  plt.ylabel('MPG')
  plt.legend()
  plt.show()

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 25]) # Should only be [0, 10] for when loss actually declines
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)
  plt.show()

# Plot the shitty guesses from the untrained model. What a doofus
# plot_horsepower(x,y)

# Sets config for model training
horsepower_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    # optimizer=DummyOptimizer(),
    loss='mean_absolute_error') #   `loss = mean(abs(y_true - y_pred), axis=-1)`


history = horsepower_model.fit(
    train_features['Horsepower'], train_labels,
    epochs=100,
    # suppress logging
    verbose=0,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
tf.print(hist.tail())

plot_loss(history)


y = horsepower_model.predict(x)
# tf.print(y)
# There is variance in the below values!! But of course, less so in 100 epochs
# For 1 epoch, y = 
# array([[-2.764],
#        [-2.73 ],
#        [-2.696],
#        [-2.662],
#        [-2.628],
#        [-2.594],
#        [-2.56 ],
#        [-2.526],
#        [-2.492],
#        [-2.458],
#        [-2.424]], dtype=float32)
# For 100 epochs, y =
# array([[39.869],
#        [39.704],
#        [39.539],
#        [39.374],
#        [39.208],
#        [39.043],
#        [38.878],
#        [38.713],
#        [38.548],
#        [38.383],
#        [38.218]], dtype=float32)




# plot_horsepower(x,y)
