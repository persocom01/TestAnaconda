# Demonstrates the use of scipy in hypothesis testing. Namely the t test.
# The t test is used on y or y and one feature, x. It is used on samples <= 30.
# For samples > 30, use the z test instead.
import scipy.stats as stats
import numpy as np

# We will use t test here to determine if the null hypothesis, that the coin is
# fair is false; fair as in it has an equal chance of returning heads or tails.
# Heads being 1 and tails being 0. We do this by using the t test on the sample
# and using 0.5 as the expected mean.
chance_of_heads = 0.5
coin = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1,
        1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
print('mean chance of heads:', np.mean(coin))
tval, pval = stats.ttest_1samp(coin, chance_of_heads)

# The null hypothesis is the default position, which we must strive to prove
# untrue. In this case, it is the position that the odds of a heads are 0.5.
if pval <= 0.05:
    print(
        f'pval = {pval}, therefore reject null hypothesis that the chance of heads is {chance_of_heads}.')
else:
    print(f'pval = {pval}, therefore unable to reject null hypothesis that the chance of heads is {chance_of_heads}.')
