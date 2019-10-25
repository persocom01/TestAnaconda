import pandas as pd

data = {'col1': [2, 1, 1, 1], 'col2': [1, 3, 2, 4]}
df = pd.DataFrame(data)

# Select a column using the column label.
print(df['col1'])
print()

# Add a column by 'selecting' and defining a new column label.
df['col3'] = [1, 2, 3, 1]
print('new col:\n', df)
print()

# Demonstrates adding columns together.
df['sum'] = df['col1'] + df['col2'] + df['col3']
print('adding cols together:\n', df)
print()

# Delete a column either using del df['column_name'] or
# pop['column_name']. The only difference is pop will also return
# the column.
print('pop:')
print(df.pop('col3'))
print()
