import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb
import sklearn.datasets as skld

iris = skld.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# The function is made flexible in only needing one argument.


def subplot_dist(df, kind='dist', cols=None, titles=None, xlabels=None, ylabels=None):
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
        if kind == 'dist':
            sb.distplot(df[col], ax=ax[i])
        # Boxplotting option.
        elif kind == 'box':
            sb.boxplot(data=df[col], ax=ax[i])
            # xticklabels will be the first letter of string if not passed as a
            # list.
            if not isinstance(col, (list, tuple)):
                ax[i].set_xticklabels([col])
            else:
                ax[i].set_xticklabels(col)

        if titles:
            ax[i].set_title(titles[i])
        if xlabels:
            ax[i].set_xlabel(xlabels[i])
        if ylabels:
            ax[i].set_ylabel(ylabels[i])


# print(df)
subplot_dist(df, kind='box')
