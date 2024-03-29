# Demonstrates reindexing and renaming in pandas.
import pandas as pd

data = {'col1': [2, 1, 1, 1], 'col2': [1, 3, 2, 4]}
df = pd.DataFrame(data)
print(df)
print()

# Pretty troublesome considering you need to pass a dictionary with all the old
# labels.
df = df.rename(
    index={0: 'one', 1: 'two', 2: 'three', 3: 'four'}, columns={'col1': 'a', 'col2': 'b'})
print('renamed:')
print(df)
print()

# Faster than the previous method for renaming columns but requires an ordered
# list of all column names. Rename is faster is only a few names in a large
# table need to be changed.
df.columns = ['A', 'B']
print('renamed2:')
print(df)
print()

# df.reset_index(self, level=None, drop=False)
# drop=True replaces the original index with the new one. If drop=False, the
# original index will be put into a new first column 'index'.
df = df.reset_index(drop=True)
print('reset index:')
print(df)
print()

# df.reindex_like(self, other, method=None, copy=True, limit=None, tolerance=None)
# method='ffill', 'bfill' or 'nearest' determines how to fill in the gaps.
# ffill will replace a row[1] with a row[0] if row 1 is empty. This will apply
# even if there are multiple empty rows, unless limit=int is used to limit the
# fill effect.
# reindex is not like rename. Any current values that do not already fit into
# the new index being adopted will be replaced by nan.
data2 = {'A': [5, 2], 'B': [3, 3]}
df2 = pd.DataFrame(data2)
df2 = df2.reindex_like(df)
print('reindex like:')
print(df2)
print()
