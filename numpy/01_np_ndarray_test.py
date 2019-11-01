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
# arr.shape returns the shape of the array.
print('shape: ', arr.shape)

# arrays can be reshaped with different dimensions (row, column, height).
# The dimensions must be valid or an error will be returned.
arr.shape = (2, 6)
print(arr.shape)
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
# + - * / , and even dot products. The arrays must contain data that can be
# added together. Typically, you want the arrays to be the same shape.
arr2.shape = (6, 2)
# @ is a special operator for ndarray objects. It does matrix multiplication,
# which is basically multiply the columns in array1 with the rows in array2.
# Matrices are used to find the expectation of E(x) by mapping a matrix of
# results with a matrix of the probability of each result.
arr3 = arr @ arr2
print('matrix multiply:\n', arr3)
print()

# If an array is made up of arrays of size x, you may apply a single array of
# size x to all the faughter arrays. This is known as broadcasting.
arr.shape = (4, 3)
arr2 = np.array([10, 10, 10])
arr3 = arr + arr2
print('broadcasting:\n', arr3)
print()

# np.require() converts the data to a specific type. It can also be used to
# change the object flags.
arr4 = np.require(arr3, dtype=float, requirements=None)
arr4 = arr4 * 2
print('require:\n', arr4)
