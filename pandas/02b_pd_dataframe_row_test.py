# Demonstrates the various attributes and methods available for use with
# DataFrame rows.
import pandas as pd

data = {
    'col1': [2, 1, 1, 1],
    'col2': [1, 3, 2, 4],
    'col3': [1, 2, 3, 1],
    'col4': ['red fox', 'red fox', 'black bear', 'arctic fox']
    }
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
    index={0: 'one', 1: 'two', 2: 'three', 3: 'four'}, columns={'col1': 'A', 'col2': 'B', 'col3': 'C', 'col4': 'D'})
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
# Remember to use brackets when using conditions.
print('row subsets:')
# All rows that are not == 3 in column B.
print('rows with col B != 3')
print(df[~(df['B'] == 3)])
# All rows where column A == 1 and column B < 4.
# To chain multiple conditions, use & for and and | for or.
print('rows with col A == 3 and col B < 4')
print(df[(df['A'] == 1) & (df['B'] < 4)])
# All rows where 3 => column B >= 1.
print('rows with col B between 1 and 3 inclusive')
print(df[df['B'].between(1, 3)])
# All rows where column D contains any of a list of strings. isin() also works
# with numbers.
print('rows with col D = arctic fox or black bear')
print(df[df['D'].isin(['arctic fox', 'black bear'])])
# All rows where column D contains the word fox.
print('rows with col D contains the word fox')
# Demonstrates ignoring case. If you give 'fox' as an argument instead, we can
# write this as df[col].str.lower().str.contains('fox'), which is said to be
# slightly faster.
print(df[df['D'].str.contains('FOX', case=False)])
print()

# df.drop(self, labels=None, axis=0, index=None, columns=None, level=None,
# inplace=False, errors='raise') deletes rows. If multiple rows have the same
# label,they will all be dropped.
df = df.drop(['four', 'five'])
print('drop:')
print(df)
print()
