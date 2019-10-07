# Demonstrates ways to generate an ndarray without an input.
import numpy as np

# np.empty(shape, dtype=float, order='C') creates an ndarray with random values
# of the specified shape and dtype,
grid = np.empty([2, 3], dtype=float)
# np.zeros(), which does the same thing but filled with zeros instead.
grid2 = np.zeros([2, 3], dtype=float)

print('empty: ', grid)
print('zeros: ', grid2)
