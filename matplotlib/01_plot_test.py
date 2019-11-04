import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)

# Matplotlib objects are composed of two parts, the figure and the axes.
# They can be created together using plt.subplots(figsize=(int, int)).
fig, ax = plt.subplots(figsize=(10, 10))

# ax.plot(x, y, fmt, data, scalex=True, scaley=True, **other_line2d_properties)
# x is optional, y is the y-axis, fmt is a special format string that comprises
# [marker][line][color]. The syntax, and the arguments for
# **other_line2d_properties can be found here:
# https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
# scalex determines if view limits are adapted to the data.
ax.plot(x, y, 'o-b')

print(plt.gca())
