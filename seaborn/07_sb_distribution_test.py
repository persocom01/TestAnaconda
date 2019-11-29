import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb
import sklearn.datasets as skds

iris = skds.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())

# sb.boxplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None,
# orient=None, color=None, palette=None, saturation=0.75, width=0.8,
# dodge=True, fliersize=5, linewidth=None, whis=1.5, notch=False, ax=None,
# **kwargs)
# orient='h' makes the boxplot horizontal.

# The function is made flexible in only needing one argument.


def subplot_dist(df, kind='dist', cols=None, titles=None, xlabels=None, ylabels=None, meanline=False, medianline=False, **kwargs):
    # Accepts all columns if they can be converted to numbers if cols argument
    # is not given.
    if not cols:
        cols = []
        if kind == 'count':
            cols = [x for x in df.select_dtypes(include='object').columns]
        else:
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                    cols.append(col)
                except ValueError:
                    pass

    # Sets number of figure rows based on number of DataFrame columns.
    if len(cols) > 4:
        ncols = 3
        nrows = int(np.ceil(len(cols)/3))
    else:
        ncols = 2
        nrows = int(np.ceil(len(cols)/2))
    # Sets figure size based on number of figure rows.
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 5*nrows))
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
                ax[i].axvline(np.mean(df[col]), color='r',
                              linestyle='-', linewidth=1)
            if medianline:
                ax[i].axvline(np.median(df[col]), color='purple',
                              linestyle='--', linewidth=1)
        # Boxplotting option.
        elif kind == 'box':
            sb.boxplot(data=df[col], ax=ax[i], **kwargs)
            if not is_list and meanline:
                ax[i].axhline(np.mean(df[col]), color='r',
                              linestyle='-', linewidth=1)
        # Countplot option for nominal variables.
        elif kind == 'count':
            sb.countplot(x=col, data=df, ax=ax[i], **kwargs)

        if titles:
            ax[i].set_title(titles[i])
        if xlabels:
            ax[i].set_xlabel(xlabels[i])
        if ylabels:
            ax[i].set_ylabel(ylabels[i])

    plt.show()
    plt.clf()


# print(df.head())
subplot_dist(df, cols=[['sepal length (cm)', 'petal length (cm)'],
                       'sepal width (cm)'], kind='box', meanline=True)
