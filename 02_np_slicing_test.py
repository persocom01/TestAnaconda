# Demonstrates indexing and slicing methods with ndarrays.
import numpy as np

a = np.arange(10, 20)
# You may slice ndarrys just like lists in python.
s = slice(0, 15, 2)
print('slice: ', a[s])

# indexing works similarly as python, however, in addition, one may
# use ... to slice columns instead of rows.
a = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print('index: ', a[..., 2])
