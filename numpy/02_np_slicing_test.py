# Demonstrates indexing and slicing methods with ndarrays.
import numpy as np

a = np.arange(10, 20)
# You may slice ndarrys just like lists in python.
s = slice(0, 15, 2)
print('slice: ', a[s])

# indexing works similarly as python, but better.
a = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
# Use ... to slice columns instead of rows.
print('... index: ', a[..., 2])
# Use start:stop to slice a range of rows or cols.
print('middle element of last two rows: ', a[-2:, 1])
# If you use two list of the same size as indexes, the first list will
# be taken as the row list and the second as the index list, the third
# as the height list etc.
# In this case, it will return the elements a list containing the elements
# [0, 0], [1, 1] and [2, 0].
print('list index: ', a[[0, 1, 2], [0, 1, 0]])
# Passing a nested list will return a nested list, which otherwise works
# the same as before.
# In this case, it will return [0, 0], [0, 2] for the first list.
rows = [[0, 0], [2, 2]]
cols = [[0, 2], [0, 2]]
print('corner elements:\n', a[rows, cols])
