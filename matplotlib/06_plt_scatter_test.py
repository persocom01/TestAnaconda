import numpy as np
import matplotlib.pyplot as plt

data = {'a': np.arange(50),
        'c': np.random.randint(0, 50, 50),
        'd': np.random.randn(50)}
data['b'] = data['a'] + 10 * np.random.randn(50)
data['d'] = np.abs(data['d']) * 100

# ax.scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, edgecolors=None, *, plotnonfinite=False, data=None, **kwargs)[source]
plt.scatter('a', 'b', c='c', s='d', data=data)
plt.xlabel('entry a')
plt.ylabel('entry b')
plt.show()
plt.clf()


# Demonstrates a function to plot multiple graphs at a time.
def subplot_scatter(df, xcols, ycols, titles=None, xlabels=None, ylabels=None):
    nrows = int(np.ceil(len(xcols)/2))
    fig, ax = plt.subplots(nrows=nrows, ncols=2, figsize=(16, 5*nrows))
    # Makes the list flat.
    ax = ax.ravel()
    for i, col in enumerate(xcols):
        ax[i].scatter(df[col], df[ycols[i]], [1, 2, 3, 4])
        if titles:
            ax[i].set_title(titles[i])
        if xlabels:
            ax[i].set_xlabel(xlabels[i])
        if ylabels:
            ax[i].set_ylabel(ylabels[i])
        # Draws a diagonal line across the graph.
        ax[i].plot([0, 1], [0, 1], transform=ax[i].transAxes,
                   color='red', linestyle='dashed', linewidth=1)


subplot_scatter(data, ['a', 'a', 'a'], ['b', 'c', 'd'])
