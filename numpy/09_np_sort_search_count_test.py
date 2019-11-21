# Demonstrates the various sorting, searching, and counting functions in numpy.
import numpy as np

a = np.array([
    [1, 2, 3],
    [4, 2, 6],
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
# the sorted indexes of the target. The indexes will have to be plugged back
# into the target list to return the sorted target.
target = ['middle', 'last', 'first']
print('lex sort:')
lexsorted = np.lexsort((target, a[2]))
print([target[i] for i in lexsorted])
print()

# np.argmax(a, axis=None, out=None)
print('arg max:')
print(np.argmax(a))
print()
