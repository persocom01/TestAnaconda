import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb
import sklearn.datasets as skds

iris = skds.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# The function is made flexible in only needing one argument.


def subplot_dist(df, kind='dist', cols=None, titles=None, xlabels=None, ylabels=None, meanline=False, medianline=False, **kwargs):
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
        if is_list and kind != 'box':
            print('distplot does not plot multiple series in one graph.')
            continue
        if kind == 'dist':
            sb.distplot(df[col], ax=ax[i], **kwargs)
            if meanline:
                ax[i].axvline(np.mean(df[col]), color='r', linestyle='-', linewidth=1)
            if medianline:
                ax[i].axvline(np.median(df[col]), color='purple', linestyle='--', linewidth=1)
        # Boxplotting option.
        elif kind == 'box':
            sb.boxplot(data=df[col], ax=ax[i], **kwargs)
            # xticklabels will be the first letter of string if not passed as a
            # list.
            if is_list:
                ax[i].set_xticklabels(col)
            else:
                ax[i].set_xticklabels([col])
                if meanline:
                    ax[i].axhline(np.mean(df[col]), color='r', linestyle='-', linewidth=1)

        if titles:
            ax[i].set_title(titles[i])
        if xlabels:
            ax[i].set_xlabel(xlabels[i])
        if ylabels:
            ax[i].set_ylabel(ylabels[i])


# print(df.head())
subplot_dist(df, meanline=True, medianline=True)
