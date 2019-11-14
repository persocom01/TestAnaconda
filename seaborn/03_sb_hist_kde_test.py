# Demonstrates histograms in seaborn.
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as skds
import seaborn as sb

# Use this command if using Jupyter notebook to plot graphs inline.
# %matplotlib inline

iris = skds.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())

# You plot a histogram by using seaborn's displot and setting kde=False.
# sb.distplot(a, bins=None, hist=True, kde=True, rug=False, fit=None,
# hist_kws=None, kde_kws=None, rug_kws=None, fit_kws=None, color=None,
# vertical=False, norm_hist=False, axlabel=None, label=None, ax=None)
sb.distplot(df['sepal length (cm)'], kde=False)
plt.show()
plt.clf()

# You plot a kernel density estimation by using seaborn's displot and setting
# hist=False.
sb.distplot(df['sepal length (cm)'], hist=False)
plt.show()
plt.close()
