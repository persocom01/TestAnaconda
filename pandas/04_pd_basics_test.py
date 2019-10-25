# Demonstrates some of pandas' basic attributes and methods.
import pandas as pd

data = {'col1': [2, 1, 1, 1], 'col2': [1, 3, 2, 4], 'col3': [1, 2, 3, 1]}
df = pd.DataFrame(data)
print(df)
print()

# T returns a transpose of the dataframe. Note that the dataframe
# itself it not changed. Also note T must be capitalized.
print('transpose:\n', df.T)
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
print('values:\n', df.values)
print()
# df.head(n=5). Returns the first n rows of the dataframe.
# df.tails does the same thing for the last rows.
print('head:\n', df.head())
print()
