import pandas as pd
import sklearn.model_selection as skms
import sklearn.preprocessing as skpp
import sklearn.compose as skc

# Use this command if using Jupyter notebook to plot graphs inline.
# %matplotlib inline

import_path = r'.\datasets\drinks.csv'
data = pd.read_csv(import_path, na_filter=False)
data.pop('country')
df = pd.DataFrame(data)
# Gets rid of any string variables.
df = pd.get_dummies(df, columns=['continent'], prefix='cont', drop_first=True)
features = [col for col in df.columns if col != 'total_litres_of_pure_alcohol']
target = 'total_litres_of_pure_alcohol'
print(df.head())

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = skms.train_test_split(X, y)

# skc.ColumnTransformer(transformers, remainder=’drop’, sparse_threshold=0.3,
# n_jobs=None, transformer_weights=None, verbose=False)
# transformers accepts a list of (trans_name, trans_function, cols) tuples.
# The cols in which the transformations are to be applied must not overlap.
# remainder='passthrough' causes the function to return all other columns that
# were not affected by the transformation instead of dropping them.
ct = skc.ColumnTransformer(
    # skpp.MinMaxScaler(feature_range=(0, 1), copy=True) scales variables on
    # range equal to the feature range.
    # copy=True means the new object will not replace the old.
    # Trying to scale a categorical variable will result in a ValueError.
    [('mms', skpp.MinMaxScaler(feature_range=(0, 10)), ['beer_servings']),
     # skpp.StandardScaler(copy=True, with_mean=True, with_std=True) scales
     # variables on a scale of +- std deviations about the mean.
     ('ss', skpp.StandardScaler(), ['spirit_servings'])],
    remainder='passthrough')
# If a warning occurs, set:
pd.options.mode.chained_assignment = None
X_train[features] = ct.fit_transform(X_train[features])
# The test set will be transformed but not fitted.
X_test[features] = ct.transform(X_test[features])
print('MinMaxScaler beer and StandardScaler spirits:')
print(X_train.head())

# skpp.Normalizer(norm=’l2’, copy=True) scales the variables such that the sum
# of all squares in the row=1. I'm not sure what this is used for.
norm = skpp.Normalizer()
X_train[features] = norm.fit_transform(X_train[features])
print('normalizer:')
print(X_train.values[:3])
print()
