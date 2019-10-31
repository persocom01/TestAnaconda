# Demonstrates some of pandas' basic attributes and methods.
import pandas as pd

data = {
    'col1': [2, 1, 1, 1],
    'col2': [1, 3, 2, 4],
    'col3': [1, 2, 3, 1],
    'cola': ['coke', 'pepsi', 'sprite', 'coke']
}
df = pd.DataFrame(data)
print(df)
print()

# T returns a transpose of the DataFrame. Note that the DataFrame itself it not
# changed. Also note T must be capitalized.
print('transpose:')
print(df.T)
print()
# This is the index and columns attributes combined into one.
print('axes:', df.axes)
print('empty?:', df.empty)
print('dimensions:', df.ndim)
# Returns the total number of cells. The size of a 4x3 table is 12.
print('size:', df.size)
print('shape:', df.shape)
print()
# Strips off the rows and indexes and returns an ndarray of lists.
print('values:')
print(df.values)
print()
# df.head(n=5). Returns the first n rows of the DataFrame.
# df.tails() does the same thing for the last rows.
print('head:')
print(df.head())
print()
# df[col_name].unique() returns a list of the unique values (num or str) in a
# DataFrame column. It is actually a series method, but it doesn't work when
# the series has a label. It is probably more useful to use on DataFrames.
print('unique:')
print(df['cola'].unique())
print()

# df.astype(dtype or dict_of_col_dtypes) returns a DataFrame or DataFrame
# column of the specified dtype.
df = df.astype({'col1': 'object', 'col2': 'object'})
print('head:')
print(df.dtypes)
print()

# pd.to_numeric(list, errors='raise') returns a 1d list of strings converted
# into int or float.
# Unconvertable strings will cause a ValueError by default, but errors='ignore'
# will ignore them and errors='coerce' will turn them into nan.
# to_datetime and to_timedelta are similar methods.
print('to_numeric:')
df['col1'] = pd.to_numeric(df['col1'])
print(df.dtypes)
print()

# df.infer_objects() returns an attempts to convert object dtypes to something
# more specific.
df = df.infer_objects()
print('infer_objects:')
print(df.dtypes)
print()
