import pandas as pd

data = {'col1': [2, 1, 1, 1], 'col2': [1, 3, 2, 4]}
df = pd.DataFrame(data)

# Select a column using the column label.
# If muliple columns need to be selected, pass df[list] instead.
print('select by column:')
print(df['col1'])
print()

# Add a column by 'selecting' and defining a new column label.
df['col3'] = [1, 2, 3, 1]
print('new col:')
print(df)
print()

# Demonstrates adding columns together.
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

# Select row using df.loc(row_name).
print('select by row:')
print(df.loc[[1, 3]])
print()

# Select row by row index using df.iloc(row_index).
# The row index is independent of the actual row label and starts
# from 0 like python ranges.
df = df.rename(
    index={0: 'one', 1: 'two', 2: 'three', 3: 'four'}, columns={'col1': 'A', 'col2': 'B', 'col3': 'C'})
print('select by row:')
print(df.iloc[[1, 3]])
print()

# Lastly, you may slice the dataframe by rows using df[start:end].
# like string slicing it will return the start to end-1 row.
print('row slicing:')
print(df[1:3])
print()

# Add rows using df.append(dataframe).
# Delete rows using df.drop(row_label). If multiple rows have the
# same label,they will all be dropped.
