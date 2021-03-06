# Demonstrates working with the basics of pyplot; the figure and axes.
import numpy as np
import matplotlib.pyplot as plt

# Use this command if using Jupyter notebook to plot graphs inline.
# %matplotlib inline

# Changes the graph style. A style reference can be found here:
# https://matplotlib.org/3.1.0/gallery/style_sheets/style_sheets_reference.html
plt.style.use('bmh')

np.random.seed(123)
x = np.arange(10)
y = [1, 2, 4, 6, 4, 4, 3, 5, 6, 7]
y2 = [7, 5, 4, 3, 5, 5, 2, 4, 6, 8]

# Demonstrates setting fontsize
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# matplotlib objects are composed of two parts, the figure and the axes.
# They can be created together using plt.subplots(figsize=(int, int)).
# It is here that fig parameters are best passed, namely:
# figsize=(float, float)
# dpi=int determines the resolution of the image. The default value is 100.
# facecolor=color which sets the color of the layer behind the graph.
# matplotlib doesn't use html colors. The acceptable color documentation can be
# found here: https://matplotlib.org/3.1.1/api/colors_api.html
# edgecolor=color of the figure outline. It's not quite visible unless you
# also set linewidth.
# linewidth=float
# ax parameters are often set after the figure is plotted, and contain many
# get or set methods. Documentation can be found here:
# https://matplotlib.org/3.1.1/api/axes_api.html#the-axes-class
fig, ax = plt.subplots(figsize=(8, 5), facecolor='gainsboro', edgecolor='r', linewidth=5)

# ax.plot(x, y, fmt, data, scalex=True, scaley=True, **other_line2d_properties)
# or plt.plot() plots a line graph.
# Aside from an actual, line graphs it can be used to plot lines in other
# graphs, such as a linear regression line in scatter plots.
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
# fig.legend() creates a legend outside the graph.
ax.legend(labels=['pies', 'muffins'], loc='best', facecolor='yellow', shadow=True)
# ax.set_title(self, label, fontdict=None, loc='center', pad=None, **kwargs)
ax.set_title('sales')
# ax.grid(self, b=None, which='major', axis='both', **kwargs)
# If no arguments are passed, turns on grid lines. Common arguments include:
# c=color or color=color
# ls=fmt or linestyle=fmt
# lw=float or linewidth=float
ax.grid()
# ax.set_xlim(self, left=None, right=None, emit=True, auto=False, *, xmin=None,
# xmax=None)
ax.set_xlim(0, 10)
# ax.yticks(ticks=None, labels=None, **kwargs)
# You can set the individual ticks instead of setting the axis limit, but it
# needs more arguments. Passing an empty list deletes all the ticks.
ax.set_yticks(range(0, 11, 2))
# ax.set_xlabel(self, xlabel, fontdict=None, labelpad=None, **kwargs)
ax.set_xlabel('week')
# ax.text(self, x, y, str, fontdict=None, **kwargs) places text at the stated
# xy coords. Accepts LaTeX math symbols by starting and ending the str with $.
# List of symbols can be found here:
# https://oeis.org/wiki/List_of_LaTeX_mathematical_symbols
ax.text(3, 6.5, r'$\Sigma\ pies=24$')

# Creates axes on the right side of the figure, for no good reason in this case.
ax2 = ax.twinx()

# Demonstrates saving the figure into a file.
# plt.gcf() gets the current figure.
# fig = plt.gcf()
# dpi=int allows figures to be saved with better resolution. This is necessary
# for larger graphs.
fig.savefig('./saved graphs/matplotlib graphs.jpg', dpi=100)


# Used to open the plot in a new window if not using Jupyter or Hydrogen.
plt.show()
# Clears the plot after use for a new figure, or the plots will overlap.
# Just for show in this case since only 1 figure was plotted.
# Use plt.cla() to clear current axes, and ax.clear() to clear specific axes.
plt.clf()

# Used to open the plot in a new window if not using Jupyter or Hydrogen.
# plt.show()
# Closes the plot window entirely.
plt.close()
