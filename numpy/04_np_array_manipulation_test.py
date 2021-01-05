# Demonstrates the various ways to manipulate an array in numpy.
import numpy as np

arr = np.array([
    [[1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12]],

    [[13, 14, 15, 16],
     [17, 18, 19, 20],
     [21, 22, 23, 24]],
])
list = arr.tolist()

# np.reshape(arr, newshape, order)
# The new shape of the arr must be able to fit all elements of the original.
print('reshape:')
# Can be called as an ndarray method:
# print(arr.reshape(2, 12))
print(np.reshape(list, (2, 12)))
print()

# Flat is method where one can recall individual elements in a 2d array as if
# it were a 1d array using the element index.
i = 5
print(f'flat[{i}]:', arr.flat[i])
print()

# ndarray.flatten(order)
# Similar to a reshape that always returns a 1d array without it being embedded
# in a list.
print('flatten:')
print(arr.flatten())
print()

# np.ravel(a, order)
# Similar to flatten, with a few key differences:
# 1. It can be used on arrays other than ndarrays.
# 2. It returns a view of the original array instead of a copy. As such,
# changes made to the ravel object may affect the original.
# 3. It is normally faster since it does not use memory.
# Like reshape, reval can be called as a ndarray method, but only on ndarrays:
# print(arr.ravel())
print('ravel:')
print(np.ravel(list))
print()

# np.transpose(arr, axes)
# Flip the array about its axes, for instance (1, 2) to (2, 1). It differs from
# reshape in that the values themselves are remapped from rows to columns and
# vice versa.
print('transpose:')
# Can be called as an ndarray method:
# print(arr.transpose())
# More commonly as a shortform:
# print(arr.T)
t = np.transpose(arr)
print(t)
print('shape:', arr.shape, 'to', t.shape)
print()

# np.rollaxis(arr, axis, start)
# axis = axis to put in first place. In the case of an array shape (2, 3, 4),
# 2, 3 and 4 are axis 0, 1 and 2 respectively. Negative nunmbers can also be
# used, for isntance, -1 refers to the last axis, or 4.
# What rollaxis does is for instance, np.rollaxis(arr, -1, 1) is to move the
# last axis (4) forward into position 1 of the shape, which will be (2, 4, 3)
# To reverse rollaxis, use np.rollaxis(arr, prev_start, prev_axis+1). prev_axis
# assumes you did not use a -ve number.
print('rollaxis:')
r = np.rollaxis(arr, 2, 1)
print(r)
print('shape:', arr.shape, 'to', r.shape)
print('return:', np.rollaxis(r, 1, 3).shape)
print()

# np.swapaxes(arr, axis1, axis2)
print('swapaxis:')
print(np.swapaxes(arr, 0, 1).shape)
# Can be called as an ndarray method:
# print(arr.swapaxes(0, 1).shape)
print()
