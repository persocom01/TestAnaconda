# Demonstrates the various sorting, searching, and counting functions in numpy.
import numpy as np

a = np.array([
    [1, 2, 3],
    [4, 2, 0],
    [7, 8, 3]
])

# np.sort(a, axis=-1, kind=None, order=None) sorts an array.
# axis=-1 sorts by the last axis by default. If set to None, it flattens the
# array before sorting.
print('sort:')
print(np.sort(a))
print()

# np.argsort(a, axis=-1, kind=None, order=None) returns the indices of the
# unsorted array that would be mapped onto the sorted array if the sort were
# performed.
print('arg sort:')
print(np.argsort(a))
print()

# np.lexsort((target, keys), axis=-1) sorts a target by the keys and returns
# the sorted indices of the target. The indices will have to be plugged back
# into the target list to return the sorted target.
target = ['middle', 'last', 'first']
print('lex sort:')
lexsorted = np.lexsort((target, a[2]))
print([target[i] for i in lexsorted])
print()

# np.argmax(a, axis=None, out=None) returns the indices of the max values along
# an axis. Note that axis=None by default instead of -1.
# np.argmin(a, axis=None, out=None) does the same thing but for min values.
print('arg max:')
print(np.argmax(a))
print()

# np.nonzero(a) returns the indices of nonzero elements in an array. If passed
# a 2d array, returns two arrays, one of the row indices, one of the column
# indices.
print('non zero:')
print(np.nonzero(a))
print()

# np.where(condition, x, y) if condition is true, return x, if false, return y.
# Can be used as a map function.
print('where:')
b = [np.where(n > 3, 'L', 'S') for n in a]
print(b)
print()

# np.extract(condition, arr) returns a list of values from arr that fulfill the
# condition.
print('extract:')
print(np.extract(a > 3, a))
print()
