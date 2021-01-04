# Demonstrates the various ways to manipulate an array in numpy.
import numpy as np

a = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])

# np.reshape(arr, newshape, order)
# The new shape of the arr must be able to fit all elements of the original.
print('reshape:')
# Can also be called as an array method:
# print(a.reshape(2, 6))
print(np.reshape(a, (2, 6)))
print()

# Flat is method where one can recall individual elements in a 2d array as if
# it were a 1d array using the element index.
i = 5
print(f'flat[{i}]:', a.flat[i])
print()

# ndarray.flatten(order)
# Similar to reshape, but always returns a 1d array without it being embedded
# as if it were a 2d array.
print('flatten:')
print(a.flatten())
print()

# np.ravel(a, order)
# Similar to flatten, with a few key differences:
# 1. It can be used on arrays other than nparrays.
# 2. It returns a view of the original array instead of a copy. As such,
# changes made to the ravel object may affect the original.
# 3. It is normally faster since it does not use new memory.
# Like reshape, reval can be called as an array method, but only on nparrays:
# print(a.ravel())
print('ravel:')
list = a.tolist()
print(np.ravel(list))
