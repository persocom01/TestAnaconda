# Demonstrates ways to generate an ndarray without an input.
import numpy as np

# np.empty(shape, dtype=float, order='C') creates an ndarray with random values
# of the specified shape and dtype,
arr = np.empty([2, 3], dtype=float)
print('empty:\n', arr)
print()

# np.zeros(), which does the same thing but filled with zeros instead.
arr = np.zeros([2, 3], dtype=float)
print('zeros:\n', arr)
print()

# np.arange(start, end, step, dtype) creates evenly spaced values within a
# given range. start defaults to zero if only a single number is given as an
# argument. The range excludes the end, just like in python range().
arr = np.arange(0, 15, 3, dtype=int)
print('arange:\n', arr)
print()

# np.linspace(start, end, steps, endpoint=True, retstep=False, dtype)
# divides the given range into the number of steps given. By default, the
# last step is always stop, unlike range().
# retstep gives you the interval value.
arr, interval = np.linspace(0, 100, 5, retstep=True, dtype=int)
print('linspace:\n', arr)
print('interval: ', interval)
print()

# np.logspace(start, end, num=50, endpoint=True, base=10, dtype)
# is like np.linspace, but the range is given in powers of base 10 by default.
arr = np.logspace(1, 10, num=10, base=2)
print('logspace:\n', arr)
print()

# Demonstrates the np.random module to generate ndarrays.
# Setting the seed makes the result of random always the same.
np.random.seed(123)
# np.random.rand(*size) generates numbers between 0 and 1.
# *size accepts any number of arguments as a new size dimension.
arr = np.random.rand(2, 3)
print('rand:\n', arr)
print()
# np.random.randint(start, stop, size) generates random integers between
# start and stop-1.
r = np.random.randint(0, 10)
print('rand int: ', r)
print()
# Arrays with more dimensions may be generated using a longer tuple for size.
arr = np.random.randint(0, 10, size=(3, 3))
print('rand int 2d array:\n', arr)
print()
# np.random.randn(*size) returns samples from a normal distribution.
arr = np.random.randn(2, 3)
print('randn:\n', arr)
print()
# Use np.random.normal if you don't just want a default (mean=0, std=1)
# normal distribution.
mean = 10
std = 2
size = (2, 3)
arr = np.random.normal(mean, std, size=size)
print('normal\n', arr)
print()
# np.random.permutation(int or arr) returns a randomly arranged list of numbers
# range(int) if int is given, or a randomly arranged list if list is given.
arr = np.random.permutation(10)
print('permutation\n', arr)
print()