# Demonstrates merging DataFrames in pandas.
# Merging is different from concat because the df are not just added to each
# other, but instead, are united into one based on a shared key.
import pandas as pd

data = {
    'id': [1, 2, 3, 4, 5],
    'name': ['apple', 'banana', 'grape', 'lemon', 'papaya'],
    'state': ['fresh', 'fresh', 'fresh', 'fresh', 'fresh']
}
data2 = {
    'id': [1, 2, 4, 5],
    'taste': ['sweet', 'sweet', 'sour', 'sweet'],
    'state': ['fresh', 'fresh', 'fresh', 'fresh'],
    'alt_id': [3, 2, 1, 5]
}
left = pd.DataFrame(data)
right = pd.DataFrame(data2)

# pd.merge(left, right, how='inner', on=none, left_on=none, right_on=none,
# left_index=False, right_index=False, sort=True)
# on=col_or_cols makes the merge based on one key or keys respectively.
# Left_on + right_on allows you to not use a common key for both DataFrames
# in the merge.
# left_index=True will use the left DataFrame's row labels as its key instead
# of a column.
# If a common key was used, the merged DataFrame will only have a single key
# column; it will not be repeated.
# Any repeated column labels are given the suffix _x for the left and _y for
# the right.
print('inner merge:')
print(pd.merge(left, right, on='id'))
print()

# how='inner' returns a DataFrame only of the keys common to both parent dfs.
# how='left', 'right' or 'outer' will take keys from one or both dfs and return
# a table where nan fills the missing spaces. Whether you use left or right
# depends on which DataFrame you want intact.
print('outer merge:')
print(pd.merge(left, right, how='outer', on='id'))
print()

# Demonstrates how to merge columns with different names.
print('merge on cols with different names:')
print(pd.merge(left, right.set_index('alt_id'), left_on='id', right_index=True))

# df.join(right, on=None, how='left', lsuffix='', rsuffix='', sort=False) is a
# faster version of merge that doesn't get rid of the shared key unless you use
# the on argument.
# Instead, you may need to specify one of the suffixes of the DataFrame's
# column label so that repeated labels will be differentiated after the merge.
# The indexes of the DataFrames are always used as the key unless the on
# argument is used, in which case it's probably better to use pd.merge().
# Best used to join DataFrames with no common keys except the index.
print('join:')
print(left.join(right, lsuffix='_left', rsuffix='_right'))
print()
