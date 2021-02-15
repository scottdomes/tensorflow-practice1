import tensorflow as tf

# This is me trying to manually build a simple neural network, step by step


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

# This data is the same as horsepower[:10] in horsepower.py
horsepower = [[ 75.], [ 88.], [160.], [ 63.], [ 67.], [ 90.], [ 60.], [ 67.], [ 95.], [ 88.]]

results = []

for horsepower_unit in horsepower:
  print(horsepower_unit, kernel_value)
  results.append([horsepower_unit[0] * kernel_value])

print(results)

# STAGE TWO: Loss function + dummy optimizer

# The below code is the equivalent of...
# horsepower_model.compile(
#     optimizer=DummyOptimizer(),
#     loss='mean_absolute_error')

# history = horsepower_model.fit(
#     train_features['Horsepower'], train_labels,
#     epochs=100, # Number of epochs really doesn't matter
#     # suppress logging
#     verbose=0)

# Which yields a loss of 23.310509
