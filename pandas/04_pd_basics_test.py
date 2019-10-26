# Demonstrates some of pandas' basic attributes and methods.
import pandas as pd

data = {'col1': [2, 1, 1, 1], 'col2': [1, 3, 2, 4], 'col3': [1, 2, 3, 1]}
df = pd.DataFrame(data)
print(df)
print()

# T returns a transpose of the DataFrame. Note that the DataFrame
# itself it not changed. Also note T must be capitalized.
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

# df.astype(dtype or dict_of_col_dtypes) is used to change the data
# type of a DataFrame or a DataFrame column.
df = df.astype({'col1': 'object', 'col2': 'object'})
print('head:')
print(df.dtypes)
print()

# pd.to_numeric(list) converts a list of strings into int or float
# and returns it. It only accepts 1d lists. Unconvertable strings
# will cause a ValueError.
# to_datetime and to_timedelta are similar methods.
print('to_numeric:')
print(pd.to_numeric(df['col1']))
print()

# df.infer_objects() Attempts to convert object dtypes to something
# more specific.
df = df.infer_objects()
print('infer_objects:')
print(df.dtypes)
print()
