import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb
import sklearn.datasets as skds

iris = skds.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# sb.pairplot(data, hue=None, hue_order=None, palette=None, vars=None,
# x_vars=None, y_vars=None, kind='scatter', diag_kind='auto', markers=None,
# height=2.5, aspect=1, dropna=True, plot_kws=None, diag_kws=None,
# grid_kws=None, size=None)
sb.pairplot(df)
