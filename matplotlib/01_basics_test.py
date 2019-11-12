# Demonstrates working with the basics of pyplot; the figure and axes.
import numpy as np
import matplotlib.pyplot as plt

# Use this command if using Jupyter notebook to plot graphs inline.
# %matplotlib inline

x = np.arange(10)
y = [1, 2, 4, 6, 4, 4, 3, 5, 6, 7]
y2 = [7, 5, 4, 3, 5, 5, 2, 4, 6, 8]

# matplotlib objects are composed of two parts, the figure and the axes.
# They can be created together using plt.subplots(figsize=(int, int)).
# It is here that fig parameters are best passed, namely:
# figsize=(float, float)
# facecolor=color which sets the color of the layer behind the graph.
# matplotlib doesn't use html colors. The acceptable color documentation can be
# found here: https://matplotlib.org/3.1.1/api/colors_api.html
# edgecolor=color of the figure outline. It's not quite visible unless you
# also set linewidth.
# linewidth=float
fig, ax = plt.subplots(figsize=(8, 6), facecolor='gainsboro', edgecolor='r', linewidth=5)

# ax.plot(x, y, fmt, data, scalex=True, scaley=True, **other_line2d_properties)
# x is optional, y is the y-axis, fmt is a special format string that comprises
# [marker][line][color].
# If the data argument is not passed, x, y, fmt can be repeated to plot
# plot multiple lines on the same graph.
# **other_line2d_properties include:
# alpha=0_to_1 for transparency.
# zorder=float to determine which graphs get to go in front. The later graphs
# appear to go in front by default.
# The syntax for fmt and the **other_line2d_properties can be found here:
# https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
# It is possible to just pass it a pandas DataFrame, however, the DataFrame
# labels will not automatically be passed as well. To show legend, use:
# ax.legend(labels=df)
ax.plot(x, y, 'o-g', x, y2, 'x-r')
# ax.legend(self, *args, **kwargs) draws a legend on the graph, and accepts a
# number of arguments, including:
# labels=list for legend labels if the data passed had no column names.
# It has to be a list. If passed a string, it will treat string the string as
# a list by using string indexes like str[0] as labels.
# loc=string which determines the location of the legend. The strings accepted
# start with the vertical position, upper, center or lower, and then the
# vertical position, left, center or right. The middle of the graph is just
# center, and best lets decides the best of the 9 possible positions based on
# minimal overlap.
# facecolor=color
# shadow=False
# Other arguments can be found here:
# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.legend.html
ax.legend(labels=['pies', 'muffins'], loc='best', facecolor='yellow', shadow=True)
