# Demonstrates plotting multiple graphs on the same figure. This is not the
# same as plotting multiple datasets on the same graph.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as skld

iris = skld.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# print(df.head())

# plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True,
# subplot_kw=None, gridspec_kw=None, **fig_kw)
# The first way to plot multiple graphs is to pass them into plt.subplots() as
# the nrow and ncols arguments.
fig, ax = plt.subplots(2, 2, figsize=(16, 10))
# Flattens the (2, 2) matrix ax is currently in.
ax = ax.ravel()
for i, col in enumerate(df.columns):
    ax[i].hist(df[col])
plt.show()
plt.clf()

# ax is automatically plt.subplot(111) if no arguments are given.
fig, ax = plt.subplots(figsize=(16, 10))
# plt.subplot(int) accepts either a 3 digit int or 3 separate ints which
# represent row, column and index of the specific subplot. Index goes from
# left to right, then up to down.
ax = plt.subplot(111)
ax.hist(df['sepal length (cm)'])
# The second way is to use fig.add_subplot(self, *args, **kwargs), which does
# not override the current figure.
ax2 = fig.add_subplot(221)
ax2.hist(df['sepal width (cm)'], color='r')
