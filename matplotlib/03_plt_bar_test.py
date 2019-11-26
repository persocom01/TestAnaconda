# Demonstrates plotting a bar chart is matplotlib.
# It is said that the most important visualization properties are position,
# color, and size.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Use this command if using Jupyter notebook to plot graphs inline.
# %matplotlib inline

data = {
    'movie': ['comedy', 'action', 'romance', 'drama', 'scifi'],
    'profit': [4, 5, 6, 1, 4],
    'cost': [7, 10, 7, 5, 5]
}
df = pd.DataFrame(data)

fig, ax = plt.subplots()
# ax.bar(self, x, height, width=0.8, bottom=None, *, align='center', data=None,
# **kwargs), or alternatively, plt.bar().
# bottom=num_list determines the lowest value on the y axis. It has the effect
# of pushing up where all the bars are drawn, although it can't be seen if it
# is the only graph being plotted.
# align can be set to 'edge' to align the label to the left side of the bar
# graph. For the right side, set width to a -ve number.
ax.bar(df['movie'], df['profit'], edgecolor='black', lw=2)
plt.show()
plt.clf()

fig, ax = plt.subplots()
# Demonstrates adding a reference vertical mean profit line to the graph.
# ax.vlines(self, x, ymin, ymax, colors='k', linestyles='solid', label='', *,
# data=None, **kwargs) produces vertical lines. For a single line, use
# ax.axvline(self, x=0, ymin=0, ymax=1, **kwargs)
ax.vlines([np.mean(df['profit']), np.mean(df['cost'])], -1,
          len(df.columns)+2, color=['r', 'g'], ls='--', lw=3)
# Demonstrates how to create a stacked bar chart.
# ax.barh(y, width, height=0.8, left=None, *, align='center', **kwargs) or
# alternatively, plt.barh().
ax.barh(df['movie'], df['profit'])
plt.barh(df['movie'], df['cost'], left=df['profit'])
plt.show()

# To do:
# Side by side bars.
# Categorical bars.
# Column graph.

plt.close()
