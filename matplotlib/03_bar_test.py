# Demonstrates the plt.bar() which is a bar graph.
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

# plt.bar(x, y, width=0.8, bottom=None, align='center', data=None, **kwargs)
# bottom=num_list determines the lowest value on the y axis. It has the effect
# of pushing up where all the bars are drawn, although it can't be seen if it
# is the only graph being plotted.
# align can be set to 'edge' to align the label to the left side of the bar
# graph. For the right side, set width to a -ve number.
plt.bar(df['movie'], df['profit'])
plt.show()
plt.clf()

fig, ax = plt.subplots()
# Demonstrates adding a reference vertical mean profit line to the graph.
# axhline produces a horizontal line.
ax.axvline(np.mean(df['profit']), color='crimson', linestyle='--', lw=3)

# Demonstrates how to create a stacked bar chart.
# plt.bar(y, x, )
plt.barh(df['movie'], df['profit'])
plt.barh(df['movie'], df['cost'], left=df['profit'])
plt.show()
plt.close()
