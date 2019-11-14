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



def subplot_histograms(dataframe, list_of_columns, list_of_titles, list_of_xlabels, ylimit=None, medianline=False):
    nrows = int(np.ceil(len(list_of_columns)/2))  # Makes sure you have enough rows
    if ylimit:
        fig, ax = plt.subplots(nrows=nrows, ncols=2, figsize=(16, 5*nrows),
                               sharey=True)  # You'll want to specify your figsize
        plt.yticks(np.arange(0, ylimit+1, np.floor((ylimit)/10)))
    else:
        fig, ax = plt.subplots(nrows=nrows, ncols=2, figsize=(16, 10)
                               )  # You'll want to specify your figsize
    ax = ax.ravel()  # Ravel turns a matrix into a vector, which is easier to iterate
    for i, column in enumerate(list_of_columns):  # Gives us an index value to get into all our lists
        # feel free to add more settings
        ax[i].hist(dataframe[column], edgecolor='black', linewidth=2)
        if medianline:
            ax[i].axvline(np.median(dataframe[column]), color='red',
                          linestyle='dashed', linewidth=1)
        # Set titles, labels, etc here for each subplot
        ax[i].set_title(list_of_titles[i])
        ax[i].set_xlabel(list_of_xlabels[i])
