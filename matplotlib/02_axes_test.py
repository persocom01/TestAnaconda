import numpy as np
import matplotlib.pyplot as plt

# plt.style.use(str_dict) determines the style of plot to use. An online
# reference can be found here:
# https://matplotlib.org/3.1.1/gallery/style_sheets/style_sheets_reference.html
# It is possible to pass your own style with a dict.
# print(plt.style.available) returns a list of accepted style strings.
plt.style.use('bmh')

# Sample rate, or the number of plot points.
fs = 500
# Frequency of signal, or the number of sin waves.
f = 2
# Determines the size of the x axis.
x = np.arange(fs)
x2 = np.arange(fs)
y = np.cos(2 * np.pi * f * (x2 / fs))
y2 = np.sin(2 * np.pi * f * (x2 / fs))

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, '-g', x2, y2, '|r')
