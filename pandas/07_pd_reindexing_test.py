# Demonstrates reindexing and renaming in pandas.
import pandas as pd

data = {'col1': [2, 1, 1, 1], 'col2': [1, 3, 2, 4]}
df = pd.DataFrame(data)
print(df)
print()

# Pretty troublesome considering you need to pass a dictionart with
# all the old labels.
df = df.rename(
    index={0: 'one', 1: 'two', 2: 'three', 3: 'four'}, columns={'col1': 'A', 'col2': 'B'})
print('renamed:')
print(df)
print()