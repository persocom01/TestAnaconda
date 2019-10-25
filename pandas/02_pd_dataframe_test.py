# A dataframe is a 2d data structure. Basically, a standard table.
# Demonstrates the various attributes and methods available for use
# with dataframes.
import pandas as pd

# The dataframe constructor is as follows:
# pd.DataFrame(data, index, columns, dtype, copy=False)
# index being the row label and columns being the column label.
# They default to a range starting from 0 if no argument is passed.
# However, if you pass pandas a dictionary, it is smart enough to use
# dict keys as column labels.
data = {'col1': [2, 1, 1, 1], 'col2': [1, 3, 2, 4], 'col3': [1, 2, 3, 1]}
df = pd.DataFrame(data)
print(df)
print()

print('shape:', df.shape)
print('columns:', df.columns)
print('index:', df.index)
print()
print(df.dtypes)
print()
# df.head(n=5). Returns the first n rows of the dataframe.
print('head:\n', df.head())
