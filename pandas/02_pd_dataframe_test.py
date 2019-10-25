# A dataframe is a 2d data structure. Basically, a standard table.
# Demonstrates the various attributes and methods available for use
# with dataframes.
import pandas as pd

# The dataframe constructor is as follows:
# pd.DataFrame(data, index, columns, dtype, copy=False)
# index being the row label and columns being the column label.
# They default to a range starting from 0 if no argument is passed.
# pd.DataFrame() creates an empty dataframe.
df = pd.DataFrame()
print(df)
print()

# If you pass pandas a dictionary, pandas is smart enough to use
# dict keys as column labels.
data = {'col1': [2, 1, 1, 1], 'col2': [1, 3, 2, 4], 'col3': [1, 2, 3, 1]}
df = pd.DataFrame(data)
print(df)
print()

# If you pass it a list of dictionaries, pandas will combine them
# arranged by column name and use NaN to fill in the missing spaces.
data = [{'col4': [2, 1, 1, 1]}, {'col2': [1, 3, 2, 4], 'col3': [1, 2, 3, 1]}]
df = pd.DataFrame(data)
print(df)
print()

# You may subsequently access the some of the arguments by using
# the dataframe attributes
print('columns:', df.columns)
print('index:', df.index)
print()
print(df.dtypes)
print()
