# A DataFrame is a 2d data structure. Basically, a standard table.
# Demonstrates the various attributes and methods available for use with
# DataFrames.
import pandas as pd
# Changes pandas default maximum displayed cols and rows.
print('max cols before:', pd.get_option('display.max_columns'))
pd.set_option('display.max_columns', 100)
print('max cols after:', pd.get_option('display.max_columns'))
print('max rows before:', pd.get_option('display.max_rows'))
pd.set_option('display.max_rows', 300)
print('max rows after:', pd.get_option('display.max_rows'))

pd.options.display.max_rows = 300

# The DataFrame constructor is as follows:
# pd.DataFrame(data, index, columns, dtype, copy=False).
# index is the row label and columns is the column label.
# They default to a range starting from 0 if no argument is passed.
# pd.DataFrame() creates an empty DataFrame.
# Lower caps DataFrame does not work.
df = pd.DataFrame()
print(df)
print()

# If you pass pandas a dictionary, pandas is smart enough to use dict keys as
# column labels.
data = {'col1': [2, 1, 1, 1], 'col2': [1, 3, 2, 4], 'col3': [1, 2, 3, 1]}
df = pd.DataFrame(data)
print(df)
print()

# If you pass it a list of dictionaries, pandas will combine them arranged by
# column name and use NaN to fill in the missing spaces.
data = [{'col4': [2, 1, 1, 1]}, {'col2': [1, 3, 2, 4], 'col3': [1, 2, 3, 1]}]
df = pd.DataFrame(data)
print(df)
print()

# You may subsequently access the some of the arguments by using the DataFrame
# attributes.
print('columns:', df.columns)
print('index:', df.index)
print()
print(df.dtypes)
print()
