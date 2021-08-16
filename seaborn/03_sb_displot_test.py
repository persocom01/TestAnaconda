# Demonstrates distribution plots in seaborn.
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as skds
import seaborn as sb

# Use this command if using Jupyter notebook to plot graphs inline.
# %matplotlib inline

iris = skds.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())

# You plot a histogram by using seaborn's displot or histplot.
# sb.displot(data=None, *, x=None, y=None, hue=None, row=None, col=None,
# weights=None, kind='hist', rug=False, rug_kws=None, log_scale=None,
# legend=True, palette=None, hue_order=None, hue_norm=None, color=None,
# col_wrap=None, row_order=None, col_order=None, height=5, aspect=1,
# facet_kws=None, **kwargs)
sb.displot(df['sepal length (cm)'], kind='hist')
plt.show()
plt.close()

# histplot largely does the same thing but but you can more easily combine it
# with a kde by setting kde=True.
# sb.histplot(data=None, *, x=None, y=None, hue=None, weights=None,
# stat='count', bins='auto', binwidth=None, binrange=None, discrete=None,
# cumulative=False, common_bins=True, common_norm=True, multiple='layer',
# element='bars', fill=True, shrink=1, kde=False, kde_kws=None, line_kws=None,
# thresh=0, pthresh=None, pmax=None, cbar=False, cbar_ax=None, cbar_kws=None,
# palette=None, hue_order=None, hue_norm=None, color=None, log_scale=None,
# legend=True, ax=None, **kwargs)
sb.histplot(df['sepal length (cm)'], kde=True)
plt.show()
plt.close()

sb.displot(df['sepal length (cm)'], kind='kde')
plt.show()
plt.close()

# kdeplot allows you to do a cumulative kernel density estimate (ckde) by
# setting cumulative=True.
# seaborn.kdeplot(x=None, *, y=None, shade=None, vertical=False, kernel=None,
# bw=None, gridsize=200, cut=3, clip=None, legend=True, cumulative=False,
# shade_lowest=None, cbar=False, cbar_ax=None, cbar_kws=None, ax=None,
# weights=None, hue=None, palette=None, hue_order=None, hue_norm=None,
# multiple='layer', common_norm=True, common_grid=False, levels=10,
# thresh=0.05, bw_method='scott', bw_adjust=1, log_scale=None, color=None,
# fill=None, data=None, data2=None, warn_singular=True, **kwargs)
sb.kdeplot(df['sepal length (cm)'], cumulative=True)
plt.show()
plt.close()

# An empirical cumulative distribution function similar to a ckde but which
# uses actual values to plot the curve, resulting in a choppy curve for small
# populations. It does not bin the data before plotting to fit a curve, and is
# therefore not an estimate like ckde.
sb.displot(df['sepal length (cm)'], kind='ecdf')
plt.show()
plt.close()

# ecdfplot allows you to plot the complement of the empirical cumulative
# distribution function by setting complementary=True.
# seaborn.ecdfplot(data=None, *, x=None, y=None, hue=None, weights=None,
# stat='proportion', complementary=False, palette=None, hue_order=None,
# hue_norm=None, log_scale=None, legend=True, ax=None, **kwargs)
sb.ecdfplot(df['sepal length (cm)'], complementary=True)
plt.show()
plt.close()
