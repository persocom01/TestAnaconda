import numpy as np
import matplotlib.pyplot as plt

plt.style.use('bmh')
# Use this command if using Jupyter notebook to plot graphs inline.
# %matplotlib inline

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, density=1, facecolor='g', alpha=0.75)

plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()


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
