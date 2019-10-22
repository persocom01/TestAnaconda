import numpy as np

a = np.array([
    [1, 2, 3],
    [4, 2, 6],
    [7, 8, 3]
])

# np.amin(arr, dimension).
print('min: ', np.amin(a, 1))
print('max: ', np.amax(a, 1))
