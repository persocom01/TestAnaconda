import pandas as pd
import sklearn.datasets as skds

# Use this command if using Jupyter notebook to plot graphs inline.
# %matplotlib inline

# sklarn.datasets comes with a few small standard datasets:
# skds.load_boston() boston house-prices dataset(regression).
# skds.load_iris() iris dataset(classification).
# skds.load_diabetes() diabetes dataset(regression).
# skds.load_digits() digits dataset(classification).
# skds.load_linnerud() linnerud dataset(multivariate regression).
# skds.load_wine() wine dataset(classification).
# skds.load_breast_cancer() breast cancer wisconsin dataset(classification).

# Each dataset object is a dictionary that contains several attributes.
data = skds.load_boston()
print(data.keys())

# The most important attributes are data, target, and feature_names, which
# are used to initialize the DataFrame and test models.
df = pd.DataFrame(data.data, columns=data.feature_names)
# Some datasets also have a target_names attribute, but it may not be the
# target column names, but rather the names for the target numerical
# categories.
target = pd.DataFrame(data.target)

print('data:')
print(df.head())
print()
print('target:')
print(target.head())
print()