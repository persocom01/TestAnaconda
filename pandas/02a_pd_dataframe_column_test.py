# Demonstrates the various attributes and methods available for use with
# DataFrame columns.
import pandas as pd

data = {
    'Col 1': [2, 1, 1, 1],
    'Col 2': [1, 3, 2, 4]
    }
df = pd.DataFrame(data)

# Common code to reduce all column names to lowercase and remove or replace
# whitespaces.
df.columns = [x.lower().replace(' ', '') for x in df.columns]
print(df)
print()

# Select a column using the column label.
# If muliple columns need to be selected, pass df[list] instead.
# You may also use the attribute df.column_name like python as well.
print('select by column:')
print(df['col1'])
# Selecting by column returns the column index, even if there is only 1 row in
# the selection. To return the actual value, use .iloc[0]. To return a list of
# values, use .tolist()
print(df['col1'].iloc[0])
print(df['col1'].tolist())
# To check if value(s) are present in a column, use isin(list) followed by
# any() or all().
print(df['col1'].isin([2]).any())
print()

# Add a column by 'selecting' and defining a new column label.
df['col3'] = [1, 2, 3, 1]
print('new col:')
print(df)
print()

# Demonstrates adding columns together.
# Addition is not the only mathematical operation available.
# For conditional mapping, use np.where:
# data['new_col'] = np.where(data['col1'] == data['col2'], value_if_true, value_if_false)
df['sum'] = df['col1'] + df['col2'] + df['col3']
print('adding cols together:')
print(df)
print()

# df.concat([df1, df2, ...], axis=0, join='outer', ignore_index=False,
# join_axes=none) can also be used to add additional columns or rows to the
# DataFrame but that will be covered in its own topic.

# There are 3 ways to delete a column.
print('del column:')
# del df['sum']
# df = df.drop('sum', axis=1)
# pop will also return the column.
print(df.pop('sum'))
print()

# Selecting a subset of columns is kind of complicated.
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
