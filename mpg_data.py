import pandas as pd

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