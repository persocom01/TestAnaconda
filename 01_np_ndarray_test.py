# The most important object in NumPy is the ndarray.
# It is similar to a python list but comes with more functions and methods.
import numpy as np

# Demonstrates the creation of an ndarray from an input.
# Alternatively, use np.asarray(), which will convert the original into an
# ndarray instead of making a new copy.
# You may also manually make a copy using np.copy()
arr = np.array([
    [[1, 2, 3],
     [4, 5, 6]],

    [[7, 8, 9],
     [10, 11, 12]],
])

# arrs can be reshaped with different dimensions (row, column, height).
# The dimensions must be valid or an error will be returned.
arr.shape = (2, 6)
print(arr)
print('type: ', type(arr))
print('dimensions: ', arr.ndim)
print('flags: ', arr.flags)
print()

arr2 = np.array([
    # int and float are compatible.
    # str is not compatible with str, however.
    [12.1, 11, 10],
    [9, 8, 7],
    [6, 5, 4],
    [3, 2, 1],
])

# Certain mathematical operations can be conducted using ndarrays.
# + - * / , and even dot products. The arrays must have the same shape and
# and contain data that can be added together.
arr2.shape = (6, 2)
# @ is a special operator for ndarray objects. It does matrix multiplication,
# which is basically multiply the rows in array1 with the columns in array2.
# This is why their matrix shapes must be 'mirrored'.
arr3 = arr @ arr2
print('matrix multiply:\n', arr3)
print()

# np.require() converts the data to a specific type. It can also be used to
# change the object flags.
arr4 = np.require(arr3, dtype=int, requirements=None)
arr4 = arr4 * 2
print('require:\n', arr4)
