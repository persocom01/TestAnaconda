# The heatmap is one of the most popular features of seaborn.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as skds
import seaborn as sb

# Use this command if using Jupyter notebook to plot graphs inline.
# %matplotlib inline

iris = skds.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())

fig, ax = plt.subplots(figsize=(12, 7.5))

# df.corr(self, method='pearson', min_periods=1)
corr = df.corr()
mask = np.array(corr)
# Sets the upper triangle above the diagonal in the array to False.
mask[np.tril_indices_from(mask)] = False
sb.set(font_scale=1)
# sb.heatmap(data, vmin=None, vmax=None, cmap=None, center=None, robust=False,
# annot=None, fmt='.2g', annot_kws=None, linewidths=0, linecolor='white',
# cbar=True, cbar_kws=None, cbar_ax=None, square=False, xticklabels='auto',
# yticklabels='auto', mask=None, ax=None, **kwargs)
# mask removes the upper triangle of the heatmap.
sb.heatmap(corr, cmap='Greens', mask=mask, annot=True, annot_kws={"size": 15}, lw=0.5)
# Corrects the heatmap for later versions of matplotlib.
bottom, top = ax.get_ylim()
ax.set_ylim(bottom+0.5, top-0.5)
plt.show()
