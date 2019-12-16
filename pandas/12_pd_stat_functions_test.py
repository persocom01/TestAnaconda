import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

# Use this command if using Jupyter notebook to plot graphs inline.
# %matplotlib inline

import_path = r'.\datasets\drinks.csv'
# Necessary in this case since 'NA'='North America' in this dataset.
data = pd.read_csv(import_path, na_filter=False)
df = pd.DataFrame(data[['beer_servings', 'spirit_servings', 'wine_servings']])
print(df.head())
print()

# pct_change(self, periods=1, fill_method='pad', limit=None, freq=None, **kwargs)
# Doesn't actually return a percentage, but the number of times the previous
# value had to be multiplied by to get the current value: (after - prev) / prev
# Also why the first row is always nan.
# I haven't figured out what periods actually does. Don't use it for now.
print(df.head().pct_change())
print()

# df.cov(self, min_periods=None)
# The covariance is the non-normalized correlation between two variables.
# It's probably not as useful as the correlation.
print(df.cov())
print()

# df.corr(self, method='pearson', min_periods=1)
# Returns a correlation matrix, which is most useful as an input of
# sb.heatmap()
# method accepts 3 possible string arguments:
# 'pearson': evaluates the linear relationship between two continuous variables.
# 'kendall': evaluates if two variables are ordered in the same way. You will
# probably seldom use this.
# 'spearman': evaluates the monotonic relationship between two continuous or
# ordinal variables. In other words, when evaluating things like a ranking
# where the difference between ranks doesn't necessarily imply that they are
# close together.
fig, ax = plt.subplots(figsize=(12, 7.5))
plt.title('alcohol servings heatmap')

# df.corr(self, method='pearson', min_periods=1)
sb.heatmap(df.corr(), cmap='PiYG', annot=True)
# Corrects the heatmap for later versions of matplotlib.
bottom, top = ax.get_ylim()
ax.set_ylim(bottom+0.5, top-0.5)
plt.show()
