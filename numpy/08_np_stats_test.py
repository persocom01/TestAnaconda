# Demonstrates the statistical functions built into in numpy.
import numpy as np

a = np.array([
    [1, 2, 3],
    [4, 2, 6],
    [7, 8, 3]
])

# np.amin(arr, dimension).
print('min: ', np.amin(a, 1))
print('max: ', np.amax(a, 1))

# np.ptp(arr, axis). axis=1 = horizontal, axis=0 = vertical as far as I can
# tell. By default it flattens the array.
# ptp returns the size of the range (max - min).
print('ptp row: ', np.ptp(a, axis=1))

# np.percentile(arr, q, axis).
print('percentile: ', np.percentile(a, 70))
# np.median(arr, axis).
print('median: ', np.median(a))
# np.mean(arr, axis).
print('mean: ', np.mean(a, axis=1))
# np.average(arr, weights) uses a weighted average instead of regular mean.
weights = [1, 2, 4]
print('average: ', np.average(a, axis=1, weights=weights))
print('standard deviation: ', np.std(a, axis=1))
print('variance: ', np.var(a, axis=1))
