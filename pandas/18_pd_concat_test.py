# Concat is different from merge or join in that it works on rows (axis=0), and
# if used on columns it never gets rid of shared keys and doesn't rename
# columns with the same label name. While merge join method is 'inner' by
# default, concat uses 'outer' join by default.
import pandas as pd

data = {
    'id': [1, 2, 3, 4, 5],
    'name': ['apple', 'banana', 'grape', 'lemon', 'papaya'],
    'state': ['fresh', 'fresh', 'fresh', 'fresh', 'fresh']
}
data2 = {
    'id': [1, 2, 4, 5],
    'name': ['apple', 'banana', 'lemon', 'papaya'],
    'taste': ['sweet', 'sweet', 'sour', 'sweet'],
    'state': ['fresh', 'fresh', 'fresh', 'fresh']
}
top = pd.DataFrame(data)
bottom = pd.DataFrame(data2)

# pd.concat(list_of_objects, axis=0, join='outer', ignore_index=False)
# ignore_index=True makes it so that the column labels are ignored in the case
# of axis=0 or the row labels are ignored if axis=1.
print('concat:')
print(pd.concat([top, bottom], axis=0, join='inner'))
print()

# df.append(self, bottom, ignore_index=False, verify_integrity=False) is a
# shortcut version of concat that only works on rows (axis=0) and join='outer'
# only.
# verify_integrity=True determines if there are repeated indices and returns
# a ValueError if found.
print('append:')
print(top.append(bottom, verify_integrity=True, sort=False))
print()
