import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
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
sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')

# plt.show()

# Shows mean, count, standard deviation for each column
train_dataset.describe().transpose()

# Show just mean and std
train_dataset.describe().transpose()[['mean', 'std']]

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
# To make our training more stable, we should normalize it
# Our goal is to have a mean of 0 and a variance of 1 for every value
# To do that, we make certain values more important, and others less
# So in this case, we'd tell our model to de-emphasize displacement
# and overemphasize cylinders, and thus they'd be given equal importance
# note: std is the square root of variance,
# variance is the average degree to which each point differs from the mean

# Create the normalizer
normalizer = preprocessing.Normalization()
# Teaches the normalizer to coerce the inputs to mean 0 std 1
normalizer.adapt(np.array(train_features))
print(train_features[:5].describe().transpose()[['mean', 'std']])
print(normalizer(train_features))
