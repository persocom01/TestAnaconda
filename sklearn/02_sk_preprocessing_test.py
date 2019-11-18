import numpy as np
import pandas as pd
import sklearn.preprocessing as skpp
import sklearn.compose as skc

import_path = r'.\datasets\drinks.csv'

data = pd.read_csv(import_path, na_filter=False)
data.pop('country')
df = pd.DataFrame(data)
# Gets rid of any string variables.
df = pd.get_dummies(df, columns=['continent'], prefix='cont', drop_first=True)
features = [col for col in df.columns if col != 'total_litres_of_pure_alcohol']
target = df['total_litres_of_pure_alcohol']
print(df.head())

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
    [('mm_scaler', skpp.MinMaxScaler(feature_range=(0, 10)), ['beer_servings']),
     # skpp.StandardScaler(copy=True, with_mean=True, with_std=True) scales
     # variables on a scale of +- std deviations about the mean.
     ('std_scaler', skpp.StandardScaler(), ['spirit_servings'])],
    remainder='passthrough')
df[features] = ct.fit_transform(df[features])
print('MinMaxScaler beer and StandardScaler spirits:')
print(df.head())

# skpp.Normalizer(norm=’l2’, copy=True) scales the variables such that the sum
# of all squares in the row=1. I'm not sure what this is used for.
normalizer = skpp.Normalizer()
df2 = pd.DataFrame(normalizer.fit_transform(df), columns=df.columns)
print('normalizer:')
print(df2[:3].values)
print()
