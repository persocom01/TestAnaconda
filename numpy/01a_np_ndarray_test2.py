# Demonstrates ways to generate an ndarray without an input.
import numpy as np

# np.empty(shape, dtype=float, order='C') creates an ndarray with random values
# of the specified shape and dtype.
# shape=[rows, columns] or int, where int is the length of the array.
arr = np.empty([2, 3], dtype=int)
print('empty:')
print(arr)
print()

# np.append(arr, values, axis=None) appends two numpy arrays together. Unlike
# python lists, they must have the same dimensionality or an error will be
# returned.
# axis=None/0/1 by default, arr and values are flattened before being appended.
# axis=0 appends a new row to the list.
arr = np.append(arr, [[1, 2, 3]], axis=0)
print('append:')
print(arr)
print()

# np.zeros(), which does the same thing but filled with zeros instead.
arr = np.zeros([2, 3], dtype=float)
print('zeros:')
print(arr)
print()

# np.arange(start, end, step, dtype) creates evenly spaced values within a
# given range. start defaults to zero if only a single number is given as an
# argument. The range excludes the end, just like in python range().
arr = np.arange(0, 15, 3, dtype=int)
print('arange:')
print(arr)
print()

# np.linspace(start, end, steps, endpoint=True, retstep=False, dtype)
# divides the given range into the number of steps given. By default, the
# last step is always stop, unlike range().
# retstep gives you the interval value.
arr, interval = np.linspace(0, 100, 5, retstep=True, dtype=int)
print('linspace:')
print(arr)
print('interval: ', interval)
print()

# np.logspace(start, end, num=50, endpoint=True, base=10, dtype)
# is like np.linspace, but the range is given in powers of base 10 by default.
arr = np.logspace(1, 10, num=10, base=2)
print('logspace:')
print(arr)
print()

# Demonstrates the np.random module to generate ndarrays.
# Setting the seed makes the result of random always the same.
np.random.seed(123)
# np.random.rand(*size) generates numbers between 0 and 1.
# *size accepts any number of arguments as a new size dimension.
arr = np.random.rand(2, 3)
print('rand:')
print(arr)
print()
# np.random.randint(start, stop, size) generates random integers between
# start and stop-1.
r = np.random.randint(0, 10)
print('rand int:', r)
print()
# Arrays with more dimensions may be generated using a longer tuple for size.
arr = np.random.randint(0, 10, size=(3, 3))
print('rand int 2d array:')
print(arr)
print()
# np.random.randn(*size) returns samples from a normal distribution.
arr = np.random.randn(2, 3)
print('randn:')
print(arr)
print()
# Use np.random.normal(mean, std, size) if you don't just want a default
# (mean=0, std=1) normal distribution.
mean = 10
std = 2
size = (2, 3)
arr = np.random.normal(mean, std, size=size)
print('normal:')
print(arr)
print()
# np.random.poisson(mean, size) takes random samples from a poisson
# distribution. A poisson distribution is that of the number of occurances of
# an event over a period of time, given a mean. Curve wise, it looks like a
# right skewed normal distribution whose skew gets less pronounced the larger
# the mean.
arr = np.random.poisson(5, size=10)
print('poisson:')
print(arr)
print()
# np.random.exponential(scale=1.0, size=None) takes random samples from an
# exponential curve.
arr = np.random.exponential(1, size=10)
print('exponential:')
print(arr)
print()
# np.random.choice(a, size=None, replace=True, p=None) returns a random list of
# values taken from a=array_int.
# size=int_tuple determines the shape of the output. If a tuple is given, the
# output will be an array of dimensions specified in the tuple.
# replace=False makes it such that the same list index cannot be picked again.
# p=list determines the probability that each value in a is picked.
arr = np.random.choice(['Kazuma', 'Aqua', 'Megumin', 'Darkness',
                        'Chris', 'Yunyun', 'Wiz', 'Kyouya'], 3, replace=False)
print('choice:')
print(arr)
print()
# np.random.permutation(int or arr) returns a randomly arranged list of numbers
# range(int) if int is given, or a randomly arranged list if list is given.
arr = np.random.permutation(10)
print('permutation:')
print(arr)
print()

# Demonstrate adding a new axis to a 1d array.
print('new axis:')
print(arr.shape)
arr = arr[np.newaxis, :]
print(arr.shape, arr)
print()
