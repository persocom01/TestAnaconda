# Demonstrates the plt.plot() which is a line graph.
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

x = np.linspace(0, 3*np.pi, 500)
y = np.sin(x**2)

# Sample rate, or the number of plot points.
fs = 500
# Frequency of signal, or the number of sin waves.
f = 2
# Determines the size of the x axis.
x2 = np.arange(fs)
y2 = np.sin(2 * np.pi * f * (x2 / fs))

# matplotlib objects are composed of two parts, the figure and the axes.
# They can be created together using plt.subplots(figsize=(int, int)).
fig, ax = plt.subplots(figsize=(10, 10))

# ax.plot(x, y, fmt, data, scalex=True, scaley=True, **other_line2d_properties)
# x is optional, y is the y-axis, fmt is a special format string that comprises
# [marker][line][color]. The syntax, and the arguments for
# **other_line2d_properties can be found here:
# https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
# scalex determines if view limits are adapted to the data.
# If the data argument is not passed, x, y, fmt can be repeated to plot
# plot multiple lines on the same graph.
ax.plot(x, y, '-g', x2/50, y2, 'xr')
# Used to open the plot in a new window if not using Jupyter or Hydrogen.
plt.show()
# Closes the plot window entirely.
plt.close()
