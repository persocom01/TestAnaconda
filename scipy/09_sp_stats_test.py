# It is said that scipy is most used for its stats module, so if
# nothing else, one should learn to use the stats module.
import scipy.stats as stats

# norm.cdf(std_deviation, loc=0, scale=1) cdf stands for cumulative
# distribution function. It returns the probability that a value will
# be less than the number of standard deviations from the mean.
# Standard deviations can be set to -ve to get the lower end of a curve.
# loc is the mean. scale is the standard deviation of the curve.
print('normal distribution:', stats.norm.cdf(
    [2, 1, 0, -1, -2], loc=0, scale=1))
print()

# uniform.cdf(values, loc=0, scale=1) is a straight diagonal line.
# loc in this case specifies the startpoint and scale the height.
print('uniform distribution:', stats.uniform.cdf(
    [2, 1, 0, -1, -2], loc=-2, scale=4))
print()
