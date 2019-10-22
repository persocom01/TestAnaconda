import numpy as np

a = np.array([
    [1, 2, 3],
    [4, 2, 6],
    [7, 8, 3]
])

# np.amin(arr, dimension).
print('min: ', np.amin(a, 1))
print('max: ', np.amax(a, 1))

# np.ptp(arr, axis). axis=1 = horizontal, axis=0 = vertical as far
# as I can tell. By default it flattens the array.
# ptp returns the size of the range (max - min).
print('ptp row: ', np.ptp(a, axis=1))
