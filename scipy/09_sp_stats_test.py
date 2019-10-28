# It is said that scipy is most used for its stats module, so if
# nothing else, one should learn to use the stats module.
import numpy as np
import scipy.stats as stats

# norm.cdf(std_deviation, loc=0, scale=1) cdf stands for cumulative
# distribution function. It returns the probability that a value will
# be less than the number of standard deviations from the mean.
# Standard deviations can be set to -ve to get the lower end of a
# curve.
# loc is the mean. scale is the standard deviation of the curve.
print('normal distribution:', stats.norm.cdf(
    [2, 1, 0, -1, -2], loc=0, scale=1))
print()

# uniform.cdf(values, loc=0, scale=1) is a straight diagonal line.
# loc in this case specifies the startpoint and scale the height.
print('uniform distribution:', stats.uniform.cdf(
    [2, 1, 0, -1, -2], loc=-2, scale=4))
print()

np.random.seed(123)
arr = np.random.normal(10, 1, 1000)

# Gives a number of common stats of an array, being size, minmax,
# mean, variance, skew and kurtosis -3.
# Skew is a measure of how frequently the values in a data set
# reside either in the high end or low end of the curve. If most of
# the values reside in the low end, it is called a positive skew,
# and vice versa.
# Kurtosis -3 is a measure of how often extreme outliers reside in
# the array.
# It is -3 because 3 is the kurtosis of a normal distribution.
print(stats.describe(arr))
# stats.gmean(arr, axis=0, dtype) returns the geometric mean, which
# is the root to the nth power of the products of n number of inputs.
# The geometric mean of [1, 2, 4] is the cuberoot of 1 x 2 x 4.
# It is said to be more appropriate than arithmetic mean in some
# cases.
print('gmean:', stats.gmean(arr))
# The harmonic mean of n elements is n divided by the sum of the reciprocals
# of said elements. The harmonic mean of [2, 2] is 2 / (1/2 + 1/2).
# Of the 3 means, hmean tends to give the lowest value.
# It is sometimes the correct mean to use, such as when finding the average
# speed of a car that has traveled d distance at different speeds.
print('hmean:', stats.hmean(arr))
# stats.kurtosis(arr, axis=0. fisher=True, bias,
# nan_policy='propagate') is already included in stats.describe(),
# but the optional parameters may be useful for getting other
# results.
# fisher=False removes the default -3 to the result that makes
# normal distribution the reference.
# nan_policy can be set to 'raise' to raise an error, or 'omit'
# to leave nan values out altogether.
print('kurtosis:', stats.kurtosis(arr, fisher=False))
# stats.mode(arr, axis=0) returns two arrays, the first being the
# actual value(s) that occur the most. The second being the number
# of times they occur.
# These can be directly accessed by using stats.mode(arr).mode and
# stats.mode(arr).count.
print('mode:', stats.mode(arr))
