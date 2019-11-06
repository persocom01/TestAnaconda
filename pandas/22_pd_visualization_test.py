# It is said that the most important visualization properties are position,
# color, and size.
import pandas as pd
import matplotlib.pyplot as plt

data = {
    'day': [1, 2, 3, 4, 5, 6],
    'temperature': [43, 53, 50, 57, 59, 67],
    'cloud cover': [30, 67, 37, 16, 28, 81]
}
df = pd.DataFrame(data)
# Line graph.
# Single variable.
# Good at displaying the trend or the change over time of a variable alone or
# with other datasets.
# df.plot(kind, x, y, ax, subplots=False, figsize, title=none, grid=False,
# legend=True, style, xlim, ylim, rot, fontsize, colormap, **kwds) is the basic
# pandas vistualization method. It will plot a line graph by default, but the
# argument kind='graph_name' can be used specify other graphs:
# 'bar', 'barh' = bar graph, horizontal bar graph
# 'hist' = histogram
# ‘box’ for boxplot
# ‘kde’ or ‘density’ for density plots
# ‘area’ for area plots
# ‘scatter’ for scatter plots
# ‘hexbin’ for hexagonal bin plots
# ‘pie’ for pie plots
# x uses the table index by default.
# y=col_list can accept multiple column names as a list to plot multiple lines.
# Alternatively, use df[col_list].plot() during plotting instead.
# ax can accept a plot object for label customization.
# subplots splits each column into its own plot. Meaning it can make graphs
# within a graph.
# style accepts a list or dictionary that styles the plot.
# xlim and ylim accept tuples of axis min and max values.
# rot sets the x axis labels to 'horizontal', 'vertical' or
# int_degrees_anticlockwise.
# colormap='str' applies a matplotlib colormap as found here:
# https://matplotlib.org/examples/color/colormaps_reference.html
# **kwds accepts other keywords to be passed to matplotlib. There is of course,
# color (in html color codes), but also stacked, which makes bars in bar graphs
# stack on top of each other, and alpha, which sets the graph transparency if
# you display them on top of each other.
fig, ax = plt.subplots()
ax.set_ylabel('Degrees Fahrenheit')
# It is possible to plot multiple
df.plot(x='day', y=['temperature', 'cloud cover'], ax=ax,
        ylim=(10, 90), rot=20, color=['sandybrown', 'teal'])

data = {'movie': ['comedy', 'action', 'romance',
                  'drama', 'scifi'], 'number': [4, 5, 6, 1, 4]}
df = pd.DataFrame(data)
# Bar chart.
# Categorical data separated by gaps between bars and larger gaps between
# categories.
# Multiple variables possible.
# Numerical data.
# Good for quick comparision.
# Use df.plot.bar() for more customization.
df.plot(kind='bar', x='movie')

# Pie chart.
# Categorical data.
# Single variable unless pie chart within pie charts. Can also be donut shaped.
# Percentage or proportional data.
# Good at showing relative proportions.
# Use df.plot.pie() for more customization.
# Commonly used after grouping data using df.groupby().
df.plot(kind='pie', y='number', labels=df['movie'])

import_path = r'.\pandas\SacramentocrimeJanuary2006.csv'
data = pd.read_csv(import_path)
df = pd.DataFrame(data)
# Scatterplot.
# Made up of many points such as to appear continuous.
# By default uses two variables. However, a third and a fourth can be added by
# using color and size of dots as parameters. If the size is used as a third
# parameter, it is called a bubble chart.
# Continuous numerical data. (or they won't be any scattering)
# Good at comparing two specific attributes of dataset(s) so as to displaying
# trends or relationships.
# s=df['size'] or int if the same size is applied to all.
# c=df['color'] or str if the same color applies to all.
df.plot(kind='scatter', x='longitude', y='latitude', s=1, c='red')

# Histogram.
# Data is divided into bins but are not categorical.
# Single variable.
# Good at displaying distribution (no gaps) of numerically finite data.
# Alternatively, use df.hist(). Hist displays the frequency of which individual
# numbers in a list of numbers occur. bin determines the number of blocks,
# range determines the total width of those blocks.
# by='col' specifies the column to group by, although it's not as often used.
df.plot(kind='hist', y='district', bins=6, range=(1, 7))

# Boxplot.
# Good for identifying outliers.
df.plot(kind='box')
