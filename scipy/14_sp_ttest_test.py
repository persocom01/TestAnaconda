# Demonstrates the use of the scipy t-test in hypothesis testing.
# The t-test is used on continuous y or binary y and one continuous feature, x.
# It is used on small samples (approx <= 30) and does not require information
# on the population variance. In practice, population variance is normally
# unknown, sample variance is taken to be population variance in larger
# samples. No hard cuttoff exists, but the z-test is recommended when that
# happens. The difference in result between the two though, is minimal.
# For multiple features, use the f-test, which is the basis of ANOVA.
# For multicategorical y, use chi2.
import numpy as np
import scipy.stats as stats
import pandas as pd

# ttest_1samp(a, popmean, axis=0, nan_policy='propagate')
# We will use the 1 sample t-test here to determine if a coin is fair (null
# hypothesis); fair as in it has an equal chance of returning heads or tails,
# heads being 1 and tails being 0. We do this by using the t-test on the sample
# and using 0.5 as the expected mean.
chance_of_heads = 0.5
coin_flips = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1,
              1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
print('mean chance of heads:', np.mean(coin_flips))
tval, pval = stats.ttest_1samp(coin_flips, chance_of_heads)

# The null hypothesis is the default position, which we must strive to prove
# untrue. In this case, it is the position that the odds of a heads are 0.5.
null = 'coin is fair'
if pval <= 0.05:
    print(
        f'pval = {pval}, therefore reject null hypothesis that {null}.')
else:
    print(
        f'pval = {pval}, therefore unable to reject null hypothesis that {null}.')
print()

# ttest_ind(a, b, axis=0, equal_var=True, nan_policy='propagate')
# The independent samples t-test or two sample t-test is used to determine if
# the means of two samples is the same (null hypothesis). Obviously, if we are
# comparing say, the weight of buses to bicycles, the null hypothesis would
# be false. How we use this test is when we wish to compare two samples which
# ought to be the same, are still the same after factor x. For instance, the
# scores of students in the same class with half given tution and the other
# half not. This t-test can be used to determine if students with tution score
# better than those without.
no_tution_scores = [84, 24, 2, 80, 94, 78, 67, 55, 70, 37, 46, 76, 99,
                    5, 47, 40, 59, 37, 73, 93, 94, 20, 22, 34, 83, 20, 47, 84, 46, 73]
tution_scores = [27, 71, 47, 40, 20, 14, 21, 92, 61, 88, 65, 80, 26,
                 11, 23, 73, 75, 54, 14, 49, 25, 82, 71, 78, 97, 47, 39, 26, 31, 34]
# In practice, the two samples are often represented as two categories in a
# DataFrame. As such, an extra step will be taken to put our test values into
# a DataFrame.
class_dic = {'scores': no_tution_scores + tution_scores,
             'tution': [0 for x in no_tution_scores] + [1 for x in tution_scores]}
df = pd.DataFrame(class_dic)
no_tution = df[df['tution'] == 0]
tution = df[df['tution'] == 1]
print('mean no tution scores:', np.mean(no_tution['scores']))
print('mean tution scores:', np.mean(tution['scores']))
ttest, pval = stats.ttest_ind(no_tution['scores'], tution['scores'])

null = 'tution had no effect'
if pval <= 0.05:
    print(
        f'pval = {pval}, therefore reject null hypothesis that {null}.')
else:
    print(
        f'pval = {pval}, therefore unable to reject null hypothesis that {null}.')
print()

# ttest_rel(a, b, axis=0, nan_policy='propagate')
# The paired sample t-test is somewhat similar to the independent samples
# t-test but instead of two seperate samples from the same population we seek
# know if there is a significant difference in the same sample before and after
# some process. Such as the same batch of students before and after tution in
# this case. It may also be applied on two samples if the second sample was
# purposefully selected to be similar to the first. In practice the paired
# sample t-test is more sensitive than the independent samples t-test.
no_tution_scores = [12, 42, 24, 70, 86, 50, 15, 71, 16, 97, 79, 3, 12,
                    14, 44, 10, 79, 94, 97, 81, 29, 41, 95, 78, 51, 11, 81, 50, 72, 73]
tution_scores = [21, 47, 23, 65, 89, 53, 24, 67, 11, 97, 85, 0, 17,
                 22, 40, 16, 81, 99, 98, 75, 35, 41, 100, 77, 54, 16, 86, 58, 78, 73]
class_dic = {'scores': no_tution_scores + tution_scores,
             'tution': [0 for x in no_tution_scores] + [1 for x in tution_scores]}
df = pd.DataFrame(class_dic)
no_tution = df[df['tution'] == 0]
tution = df[df['tution'] == 1]
print('mean no tution scores:', np.mean(no_tution['scores']))
print('mean tution scores:', np.mean(tution['scores']))
ttest, pval = stats.ttest_ind(no_tution['scores'], tution['scores'])

null = 'tution had no effect'
if pval <= 0.05:
    print(
        f'pval = {pval}, therefore reject null hypothesis that {null}.')
else:
    print(
        f'pval = {pval}, therefore unable to reject null hypothesis that {null}.')
print()
