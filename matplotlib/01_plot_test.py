# Demonstrates the plt.plot() which is a line graph.
# Aside from plotting line graphs by itself, plot is also useful for plotting
# lines on other types of graph, such as a linear regression line for scatter
# plots.
# It is said that the most important visualization properties are position,
# color, and size.
import matplotlib.pyplot as plt
import numpy as np

# plt.style.use(str_dict) determines the style of plot to use. An online
# reference can be found here:
# https://matplotlib.org/3.1.1/gallery/style_sheets/style_sheets_reference.html
# It is possible to pass your own style with a dict.
# print(plt.style.available) returns a list of accepted style strings.
plt.style.use('bmh')
# Use this command if using Jupyter notebook to plot graphs inline.
# %matplotlib inline

# Sample rate, or the number of plot points.
fs = 500
# Frequency of signal, or the number of sin waves.
f = 2
x = np.arange(fs)
# Determines the size of the x axis.
x2 = np.arange(fs)
y = np.cos(2 * np.pi * f * (x2 / fs))
y2 = np.sin(2 * np.pi * f * (x2 / fs))

# matplotlib objects are composed of two parts, the figure and the axes.
# They can be created together using plt.subplots(figsize=(int, int)).
fig, ax = plt.subplots(figsize=(8, 6))

# ax.plot(x, y, fmt, data, scalex=True, scaley=True, **other_line2d_properties)
# x is optional, y is the y-axis, fmt is a special format string that comprises
# [marker][line][color]. The syntax, and the arguments for
# **other_line2d_properties can be found here:
# https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
# scalex determines if view limits are adapted to the data.
# If the data argument is not passed, x, y, fmt can be repeated to plot
# plot multiple lines on the same graph.
ax.plot(x, y, '-g', x2, y2, '|r')

# Demonstrates saving the figure into a file.
# plt.gcf() gets the current figure.
fig = plt.gcf()
fig.savefig('./saved graphs/matplotlib line graph.jpg')

# Used to open the plot in a new window if not using Jupyter or Hydrogen.
plt.show()
# Clears the plot after use for a new figure. Plots will overlap otherwise.
# Use plt.cla() to clear current axes, and ax.clear() to clear specific axes.
plt.clf()

# Demonstrates plotting the same lines on two different graphs instead.
plt.subplots(figsize=(10, 7))
# plt.subplot(int) accepts either a 3 digit int or 3 separate ints which
# represent row, column and index of the specific subplot. Index goes from
# left to right, then up to down.
ax = plt.subplot(211)
ax2 = plt.subplot(212, sharey=ax, sharex=ax)

ax.plot(x, y, '-g')
ax2.plot(x2, y2, '|r')

plt.show()
# Closes the plot window entirely.
plt.close()
