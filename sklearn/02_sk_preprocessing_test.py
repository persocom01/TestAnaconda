# Demonstrates various data preprocessing methods.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OrdinalEncoder

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

le = LabelEncoder()
X_le = X
le_labels = le.fit_transform(X_le.columns)
le_dict = { k:v for (k,v) in zip(X_le.columns, le_labels) }
X_le.columns = le_labels
print('LabelEncoder:', le_dict)
print(X_le.head())

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


def ordinal_scale(df, mapping=None, start_num=0):
    '''
    A convenience mapping function that accepts a DataFrame and returns it with
    each column defined as keys in the mapping dictionary mapped to its values.
    '''
    if mapping:
        cols = mapping.keys()
        for col in cols:
            df[col] = df[col].map({k: i+start_num for i, k in enumerate(mapping[col])})
            if df[col].isnull().sum() > 0:
                print(f'WARNING: not all values in column "{col}" were mapped.')
    else:
        ord = OrdinalEncoder()
        try:
            cols = df.columns
            df[cols] = ord.fit_transform(df[cols])
        except AttributeError:
            df = ord.fit_transform([df]).ravel()
    return df


data = {
    'name': ['apple', 'banana', 'orange', 'apple', 'orange'],
    'origin': ['usa', 'brazil', 'china', 'china', 'australia'],
    'supply': [361, 851, 418, 439, 279],
    'demand': [264, 833, 516, 371, 194]
}
df2 = pd.DataFrame(data)
m = {'name': ['apple', 'orange'], 'origin': ['australia', 'usa', 'brazil', 'china']}
ordinal_scale(df2, m)
print('ordinal encode:')
print(df2)
print()
