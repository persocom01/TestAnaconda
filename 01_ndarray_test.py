# The most important object in NumPy is the ndarray.
# It is similar to a python list but comes with more functions and methods.
import numpy as np

grid = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
])

print(grid)
print(type(grid))
print()

# grids can be reshaped with different dimensions. The dimensions must be valid
# or an error will be returned.
grid.shape = (6, 2)
print(grid)
