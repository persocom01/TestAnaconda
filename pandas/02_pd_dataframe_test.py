# A DataFrame is a 2d data structure. Basically, a standard table.
# Demonstrates the various attributes and methods available for use with
# DataFrames.
import pandas as pd
import numpy as np
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
data = {
    'col1': [2, 1, 1, 4, 0, np.nan],
    'col2': [1, 3, 2, 4, 0, 4],
    'col3': [1, 2, 3, np.nan, 0, 4]
    }
df = pd.DataFrame(data)
print(df)
print()

# Demonstrates assigning a value to a column
df['col4'] = 'value'
# However, this does not work on lists, so a list comprehension has to be
# performed
df['empty_dict'] = [{'key': 'value'} for e in range(len(df))]
print(df)
print()

# DataFrame.any(axis=0, bool_only=None, skipna=True, level=None, **kwargs)
# A method used to check if there are any values in a row, column, or the whole
# DataFrame.
# axis=0 checks the column downwards by default. Set to 1 or columns to check
# horizontally. None checks the whole DataFrame.
# bool_only=None by default, certain values are considered as False, such as 0,
# null or False. Set this to True to only check columns with boolean dtype.
# skipna=True considers null values to be False.
# level=int_or_level_name determines the level to check for MultiIndex axes.
print('any:')
print(df.any(axis='columns'))
print()

# DataFrame.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
# A method frequently used in data cleaning, drops any row with na values. If
# subset is given, only drops rows with na in that column(s).
print('drop null:')
print(df.dropna(subset=['col3']))
print()

# DataFrame.fillna(value=None, method=None, axis=None, inplace=False,
# limit=None, downcast=None)
# value=None determines the value used to replace null values in the DataFrame.
# A column can be passed instead to fill nulls with rows from the column.
# method=None fills nulls with values in the same column. 'pad' or 'ffill'
# fills from top to bottom. 'backfill' or 'bfill' fills from bottom to top.
# axis=0 an argument used with method to determine if the value used to fill
# the null is from the same column or from the same row. 1 switches to row.
print('fill na:')
df['col1'] = df['col1'].fillna(df['col2'])
print(df)
print()

# DataFrame.duplicated(subset=None, keep='first')
# Used to find duplicated rows.
# subset=list_of_columns
# keep=first/last/False determines which duplicate is considered false so it
# can be kept. False causes all duplicates to return True.
print('duplicated:')
print(df.duplicated(['col1', 'col2'], keep=False))
print()

# DataFrame.drop_duplicates(subset=None, keep='first', inplace=False,
# ignore_index=False)
# Drops duplicate rows in a dataframe. Useful in data preprocessing.
# subset=None can be given a list of columns to count duplicates in. Otherwise,
# all columns except the index are considered.
# keep='first' determines which duplicate to keep when one is found. Either the
# first or the last.
print('drop duplicates:')
print(df.drop_duplicates(subset=['col1'], keep='last'))
print()

# If you pass it a list of dictionaries, pandas will combine them arranged by
# column name and use NaN to fill in the missing spaces.
data = [{'col4': [2, 1, 1, 1]}, {'col2': [1, 3, 2, 4], 'col3': [1, 2, 3, np.nan]}]
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
