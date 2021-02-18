# STAGE ONE: Blind prediction
# The below code is the equivalent of...
# horsepower_model = tf.keras.Sequential([
#     layers.Dense(units=1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1))
# ])
# output = horsepower_model.predict(horsepower[:10])

# Use a set seed so I can compare my manual results consistently
# Not going to manually calculate the kernel, that's unnecessary
# initializer = tf.keras.initializers.GlorotUniform(seed=1)
# kernel = initializer(shape=(1, 1))
# tf.print(kernel)
# [[1.31112158]]
kernel_value = 1.31112158

# This data is the same as horsepower[:10]
horsepower_data = [[ 75.], [ 88.], [160.], [ 63.], [ 67.], [ 90.], [ 60.], [ 67.], [ 95.], [ 88.]]

results = []

for horsepower_unit in horsepower_data:
  # print(horsepower_unit, kernel_value)
  results.append([horsepower_unit[0] * kernel_value])

print(results)
# [[98.3341185], [115.37869904], [209.7794528], [82.60065954], [87.84514586], [118.0009422], [78.66729480000001], [87.84514586], [124.5565501], [115.37869904]]
