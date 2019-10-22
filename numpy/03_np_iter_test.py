# numpy contains an iterative method capped numpay.nditer.
# It's supposed to be more efficient than the python default,
# while having more functions.
import numpy as np

fruits = np.array([['apple', 'banana', 'orange'],
                   ['avocado', 'cherry', 'pear']])
indexes = np.array([1, 2, 3])

# One difference is np.nditer works on broadcastable arrays.
# python only repeats indexes once.
for index, fruit in np.nditer([indexes, fruits]):
    print(index, fruit)
