# Demonstrates the use of the statsmodels z-test in hypothesis testing.
import numpy as np
import pandas as pd
from statsmodels.stats import weightstats as wstats

# statsmodels.stats.weightstats.ztest(x1, x2=None, value=0,
# alternative='two-sided', usevar='pooled', ddof=1.0)
# We will use the 1 sample z-test here to determine if a coin is fair (null
# hypothesis); fair as in it has an equal chance of returning heads or tails,
# heads being 1 and tails being 0. We do this by using the z-test on the sample
# and using 0.5 as the expected mean.
chance_of_heads = 0.5
coin_flips = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1,
              1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
print('mean chance of heads:', np.mean(coin_flips))
tval, pval = wstats.ztest(coin_flips, value=chance_of_heads)

# The null hypothesis is the default position, which we must strive to prove
# untrue. In this case, it is the position that the odds of a heads are 0.5.
# t-test pval = 0.2807046476465548
null = 'coin is fair'
if pval <= 0.05:
    print(
        f'pval = {pval}, therefore reject null hypothesis that {null}.')
else:
    print(
        f'pval = {pval}, therefore unable to reject null hypothesis that {null}.')
print()

# Demonstrates the z-test on two independent samples.
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
ttest, pval = wstats.ztest(no_tution['scores'], tution['scores'])

# t-test pval = 0.3288818849529497
null = 'tution had no effect'
if pval <= 0.05:
    print(
        f'pval = {pval}, therefore reject null hypothesis that {null}.')
else:
    print(
        f'pval = {pval}, therefore unable to reject null hypothesis that {null}.')
print()
