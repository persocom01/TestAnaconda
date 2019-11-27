import pandas as pd
from sklearn.datasets import load_boston
# from sklearn.datasets import load_iris
# from sklearn.datasets import load_diabetes
# from sklearn.datasets import load_digits
# from sklearn.datasets import load_linnerud
# from sklearn.datasets import load_wine
# from sklearn.datasets import load_breast_cancer

# Use this command if using Jupyter notebook to plot graphs inline.
# %matplotlib inline

# sklarn.datasets comes with a few small standard datasets:
# load_boston() boston house-prices dataset(regression).
# load_iris() iris dataset(classification).
# load_diabetes() diabetes dataset(regression).
# load_digits() digits dataset(classification).
# load_linnerud() linnerud dataset(multivariate regression).
# load_wine() wine dataset(classification).
# load_breast_cancer() breast cancer wisconsin dataset(classification).

# Each dataset object is a dictionary that contains several attributes.
data = load_boston()
print(data.keys())

# The most important attributes are data, target, and feature_names, which
# are used to initialize the DataFrame and test models.
df = pd.DataFrame(data.data, columns=data.feature_names)
df.columns = [x.lower().replace(' ', '_') for x in df.columns]
# Some datasets also have a target_names attribute, but it may not be the
# target column names, but rather the names for the target numerical
# categories.
target = 'target'
# Not needed for datasets where the target and features are not separated.
features = [col for col in df.columns if col != target]
df[target] = pd.DataFrame(data.target)

X = df[features]
y = df[target]

print('data:')
print(X.head())
print()
print('target:')
print(y.head())
print()
