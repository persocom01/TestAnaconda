import pandas as pd

data = {'col1': [2, 1, 1, 1], 'col2': [1, 3, 2, 4]}
df = pd.DataFrame(data)

# Select a column using the column label.
# If muliple columns need to be selected, pass df[list] instead.
# You may also use the attribute df.column_name like python as well.
print('select by column:')
print(df['col1'])
print()

# Add a column by 'selecting' and defining a new column label.
df['col3'] = [1, 2, 3, 1]
print('new col:')
print(df)
print()

# Demonstrates adding columns together.
# Addition is not the only mathematical operation available.
df['sum'] = df['col1'] + df['col2'] + df['col3']
print('adding cols together:')
print(df)
print()

# Delete a column either using del df['column_name'] or
# pop['column_name']. The only difference is pop will also return
# the column.
print('pop:')
print(df.pop('sum'))
print()

# Select row using df.loc(row_name, col_name).
# Use col_name if you need to select a particular cell cells from
# particular columns.
print('select by row name:')
print(df.loc[[1, 3], ['col1', 'col2']])
print()

# Select row by row index using df.iloc(row_index, col_index).
# The row index is independent of the actual row label and starts
# from 0 like python ranges.
# You can pass : as the row_index argument if you want to split the
# dataframe by column.
df = df.rename(
    index={0: 'one', 1: 'two', 2: 'three', 3: 'four'}, columns={'col1': 'A', 'col2': 'B', 'col3': 'C'})
print('select by row index:')
print(df.iloc[1:3])
print()

# Lastly, you may slice the dataframe by rows using df[start:end].
# like string slicing it will return the start to end-1 row.
print('row slicing:')
print(df[1:3])
print()

# A subset of a dataframe may be selected by including a boolean
# condition inside df[].
print('row subset:')
print(df[df['B'] < 3])
print()

# A subset of columns instead of rows, however, is far more
# complicated.
# The following is a self-made solution:
# print('convert undesirable values into nan:')
# print(df[df.iloc[:4] < 3])
# print()
# print('remove nan columns:')
# print(df[df.iloc[:4] < 3].dropna(axis=1))
# print()

# The above assumes the dataset is free from nan to begin with.
# Stackoverflow sugested a different method:
# df.loc[:, (df != value).any(axis=0)]
# df.loc[:, (df == value).all(axis=0)]
print('column subset (Stackoverflow solution):')
print(df.loc[:, (df < 3).all(axis=0)])
print()

# Add rows using df.append(dataframe).
# The rows might need to have the same column label for this to work.
# Delete rows using df.drop(row_label). If multiple rows have the
# same label,they will all be dropped.
