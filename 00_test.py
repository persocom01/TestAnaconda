# The most important object in NumPy is the ndarray.
# It is similar to a python list but comes with more functions and methods.
import numpy as np

# Demonstrates the creation of an ndarray.
# Alternatively, use np.asarray(), which will convert the original into an
# ndarray instead of making a new copy. Other functions exist, but are all
# pretty much variants of np.array().
# There is also np.empty(shape, dtype=float, order='C'), which creates
# an ndarray with random values of the specified shape and dtype,
# and np.zeros(), which does the same thing but filled with zeros instead.
grid = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]

# grids can be reshaped with different dimensions (row, column, height).
# The dimensions must be valid or an error will be returned.
# print(grid)
# print('type: ', type(grid))
# print('dimensions: ', grid.ndim)
# print('flags: ', grid.flags)
# print()

grid2 = [
    # int and float are compatible.
    # str is not compatible with str, however.
    [12.1, 11, 10],
    [9, 8, 7],
    [6, 5, 4],
]

# Certain mathematical operations can be conducted using ndarrays.
# + - * / , and even dot products. The arrays must have the same shape and
# and contain data that can be added together.
# grid3 = grid @ grid2
grid3 = grid @ grid2
print(grid3)
