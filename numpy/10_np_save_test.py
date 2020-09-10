# Demonstrates loading and saving in numpy.
import numpy as np

a = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

file = './datasets/np_save.npy'

np.save(file, a)

a = np.load(file)

print(a)
