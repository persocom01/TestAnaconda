# Demonstrates merging DataFrames in pandas.
# Merging is different from concat because the df are not just
# added to each other, but instead, are united into one based on
# a shared key.
import pandas as pd

data = {
    'id': [1, 2, 3, 4, 5],
    'name': ['apple', 'banana', 'grape', 'lemon', 'papaya'],
    'state': ['fresh', 'fresh', 'fresh', 'fresh', 'fresh']
}
data2 = {
    'id': [1, 2, 4, 5],
    'taste': ['sweet', 'sweet', 'sour', 'sweet'],
    'state': ['fresh', 'fresh', 'fresh', 'fresh']
}
left = pd.DataFrame(data)
right = pd.DataFrame(data2)

# pd.merge(left, right, how='inner', on=none, left_on=none,
# right_on=none,left_index=False, right_index=False, sort=True)
# left_index uses the left DataFrame's row labels as keys instead
# of a column.
# on can be given a list to merge based on more than one key.
# Note how the key, which was present in both DataFrames, is not
# repeated.
# Any repeated column labels are given the suffix _x for the left
# and _y for the right.
print('inner merge:')
print(pd.merge(left, right, on='id'))
print()

# how='inner' returns a DataFrame only of the keys common to both
# parent dfs. how='left', 'right' or 'outer' will take keys from
# one or both dfs and return a table where nan fills the missing
# spaces.
print('outer merge:')
print(pd.merge(left, right, how='outer', on='id'))
print()

# df.join(right, on=None, how='left', lsuffix='', rsuffix='',
# sort=False) is a faster version of merge that doesn't get
# rid of the shared key unless you use the on argument.
# Instead, you may need to specify one of the suffixes of the
# DataFrame's column label so that repeated that will be
# differentiated after the merger.
# The indexes of the DataFrames are always used as the key unless
# the on argument is used, in which case it's probably better to
# use pd.merge().
# Best used to join DataFrames with no common keys except the
# index.
print('join:')
print(left.join(right, lsuffix='_left', rsuffix='_right'))
print()
