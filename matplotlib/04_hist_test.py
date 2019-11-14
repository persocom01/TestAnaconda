import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as skld

# Use this command if using Jupyter notebook to plot graphs inline.
# %matplotlib inline

iris = skld.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())

# ax.hist(self, x, bins=None, range=None, density=None, weights=None,
# cumulative=False, bottom=None, histtype='bar', align='mid',
# orientation='vertical', rwidth=None, log=False, color=None, label=None,
# stacked=False, *, data=None, **kwargs)
# bins determines the number of bars. The bins returned is a list of positions
# of the left edge of each bar, ending with the right edge of the last bar.
# density=True makes the graph a probability density chart.
# cumulative=True makes the value of each bin itself plus all bins before it.
# histtype='step' makes the image a single collection of bars instead of
# individual bars. It's not obvious unless you also set edgecolor.
# n = height of each bar.
# patches = the images used to create each bar.
n, bins, patches = plt.hist(df['sepal length (cm)'], histtype='stepfilled', edgecolor='k', lw=2)
plt.show()
plt.clf()

# Auto subplot histogram function to be copy pasted anywhere.


def subplot_hist(df, cols=None, titles=None, xlabels=None, ylabels=None, meanline=False, medianline=False, **kwargs):
    # Accepts all columns if they can be converted to numbers if cols argument
    # is not given.
    if not cols:
        cols = []
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
                cols.append(col)
            except ValueError:
                pass

    # Sets number of figure rows based on number of DataFrame columns.
    nrows = int(np.ceil(len(cols)/2))
    # Sets figure size based on number of figure rows.
    fig, ax = plt.subplots(nrows=nrows, ncols=2, figsize=(16, 5*nrows))
    # Makes the list of lists flat.
    ax = ax.ravel()

    for i, col in enumerate(cols):
        is_list = isinstance(col, (list, tuple))
        if is_list:
            for c in col:
                ax[i].hist(df[c], **kwargs)
        else:
            ax[i].hist(df[col], **kwargs)
            if meanline:
                ax[i].axvline(np.mean(df[col]), color='r', linestyle='-', linewidth=1)
            if medianline:
                ax[i].axvline(np.median(df[col]), color='purple', linestyle='--', linewidth=1)
        if titles:
            ax[i].set_title(titles[i])
        if xlabels:
            ax[i].set_xlabel(xlabels[i])
        if ylabels:
            ax[i].set_ylabel(ylabels[i])


subplot_hist(df, [['sepal length (cm)', 'sepal width (cm)'],
                  'petal length (cm)'], edgecolor='k', lw=2)
