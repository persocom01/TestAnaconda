# Demonstrates the various attributes and methods available for use with
# DataFrame rows.
import pandas as pd

data = {'col1': [2, 1, 1, 1], 'col2': [1, 3, 2, 4], 'col3': [1, 2, 3, 1]}
df = pd.DataFrame(data)
print(df)
print()

# Select row using df.loc(row_name, col_name).
# Use col_name if you need to select a particular cell cells from particular
# columns.
print('select by row name:')
print(df.loc[[1, 3], ['col1', 'col2']])
# It is possible to slice the df by label names.
print(df.loc[:3, :'col2'])
print()

# Select row by row index using df.iloc(row_index, col_index).
# The row index is independent of the actual row label and starts from 0 like
# python ranges.
# You can pass : as the row_index argument if you want to split the DataFrame
# by column.
df = df.rename(
    index={0: 'one', 1: 'two', 2: 'three', 3: 'four'}, columns={'col1': 'A', 'col2': 'B', 'col3': 'C'})
print('select by row index:')
print(df.iloc[1:3])
print()

# Lastly, you may slice the DataFrame by rows using df[start:end].
# like string slicing it will return the start to end-1 row.
print('row slicing:')
print(df[1:3])
print()

# A subset of a DataFrame may be selected by including a boolean condition
# inside df[].
# It should be noted that pandas does not use the standard python logical
# operators. Instead:
# & = and
# | = or
# ~ = not
print('row subsets:')
# All rows that are not == 3 in column 'B'.
print(df[~(df['B'] == 3)])
# For strings, use df[col_name].isin('str1', 'str2') instead.
print(df[df['B'].between(1, 3)])
print()

# df.append(self, other, ignore_index=False, verify_integrity=False, sort=None)
# adds rows.
# sort=True might be changed to sort=False in future versions, so it's best to
# specify.
# The new rows will automatically be allocated to their respective columns if
# column labels are provided. Otherwise, they are put into new columns.
# Any columns or rows not filled in will have value nan.
data2 = {'B': [3, 2], 'A': [2, 3]}
df2 = pd.DataFrame(data2, index=['five', 'five'])
df = df.append(df2, sort=False)
print('append:')
print(df)
print()

# df.drop(self, labels=None, axis=0, index=None, columns=None, level=None,
# inplace=False, errors='raise') deletes rows. If multiple rows have the same
# label,they will all be dropped.
df = df.drop(['four', 'five'])
print('drop:')
print(df)
print()
