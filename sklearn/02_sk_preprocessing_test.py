# Demonstrates various data preprocessing methods.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import pleiades as ple

# Use this command if using Jupyter notebook to plot graphs inline.
# %matplotlib inline

import_path = r'./datasets/drinks.csv'
data = pd.read_csv(import_path, index_col=None, na_filter=False)
data.pop('country')
df = pd.DataFrame(data)
# Gets rid of any string variables.
df = pd.get_dummies(df, columns=['continent'], prefix='cont', drop_first=True)
df.columns = [x.lower().replace(' ', '_') for x in df.columns]
target = 'total_litres_of_pure_alcohol'
features = [col for col in df.columns if col != target]
print(df.head())
print()

X = df[features]
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y)

# LabelEncoder(list) takes a list and encodes unique values as integers. The
# classes can be found using LabelEncoder().classes_. More likely to be used
# for y values, but in this case y is continuous so it is demonstrated on
# dataFrame columns instead.
le = LabelEncoder()
X_le = X
le_labels = le.fit_transform(X_le.columns)
X_le.columns = le_labels
print('LabelEncoder:', le.classes_)
print(X_le.head())
print()

# ColumnTransformer(transformers, remainder='drop', sparse_threshold=0.3,
# n_jobs=None, transformer_weights=None, verbose=False)
# transformers accepts a list of (trans_name, trans_function, cols) tuples.
# The cols in which the transformations are to be applied must not overlap.
# remainder='passthrough' causes the function to return all other columns that
# were not affected by the transformation instead of dropping them.
ct = ColumnTransformer(
    # MinMaxScaler(feature_range=(0, 1), copy=True) scales variables on
    # range equal to the feature range.
    # copy=True means the new object will not replace the old.
    # Trying to scale a categorical variable will result in a ValueError.
    [('mms', MinMaxScaler(feature_range=(0, 10)), ['beer_servings']),
     # StandardScaler(copy=True, with_mean=True, with_std=True) scales
     # variables on a scale of +- std deviations about the mean.
     # with_mean=False is needed to scale sparse matrices.
     ('ss', StandardScaler(), ['spirit_servings'])],
    remainder='passthrough')
# A warning occurs if you try an overwrite the original DataFrame like:
# X_train['features'] = ct.fit_transform(X_train)
# To make it go away set:
# pd.options.mode.chained_assignment = None
X_train = pd.DataFrame(ct.fit_transform(X_train), columns=features)
# The test set will be transformed but not fitted.
X_test = pd.DataFrame(ct.transform(X_test), columns=features)
print('MinMaxScaler beer and StandardScaler spirits:')
print(X_train.head())

# Normalizer(norm=’l2’, copy=True) scales the variables such that the sum
# of all squares in the row=1. I'm not sure what this is used for.
norm = Normalizer()
X_train = pd.DataFrame(norm.fit_transform(X_train), columns=features)
X_test = pd.DataFrame(norm.transform(X_test), columns=features)
print('normalizer:')
print(X_train.values[:3])
print()

nabe = ple.Nabe()

data = {
    'name': ['apple', 'banana', 'orange', 'apple', 'orange'],
    'origin': ['usa', 'brazil', 'china', 'china', 'australia'],
    'supply': [361, 851, 418, 439, 279],
    'demand': [264, 833, 516, 371, 194]
}
df2 = pd.DataFrame(data)
print(df2)
m = {'name': ['apple', 'orange'], 'origin': ['australia', 'usa', 'brazil', 'china']}
nabe.ordinal_scale(df2, mapping=m)
print('ordinal encode:')
print(df2)
print()
