# It is said that the most important visualization properties are position,
# color, and size.
import pandas as pd
import matplotlib.pyplot as plt

# As pandas is based on matplotlib.pyplot, you can use change the plot style
# the same way:
# plt.style.use('ggplot')
# Use this command if using Jupyter notebook to plot graphs inline.
# %matplotlib inline

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
# **kwds accepts other keywords to be passed to matplotlib such as:
# color sets graph color.
# stacked=True_False sets whether bars in bar graphs stack.
# alpha=0-1 sets the graph transparency.
# edgecolor sets the outline color of graphs.
# subplots=True lets you plot more than 1 graph at a time.
# layout=(row, col) sets the grid the subplots will be on.
# sharex or sharey=True makes subplots share the same x or y axis respectively.

data = {
    'day': [1, 2, 3, 4, 5, 6],
    'fish': [43, 53, 50, 57, 59, 67],
    'bread': [30, 67, 37, 46, 28, 51]
}
df = pd.DataFrame(data)
# Line graph.
# Single variable. Plotting with more variables is possible, but they must all
# have the same scale.
# Good at displaying the change in a variable over time of a or against other
# datasets.
fig, ax = plt.subplots()
ax.set_ylabel('Number')
df.plot(x='day', y=['fish', 'bread'], ax=ax,
        ylim=(10, 90), rot=20, color=['teal', 'sandybrown'])

# Demonstrates saving the figure into a file.
fig = plt.gcf()
fig.savefig('./saved graphs/pandas line graph.jpg')

# Area graph.
# Single variable from different datasets or multiple variables with the same
# scale.
# Good at displaying the composition of a total into its component sum
# variables.
# df.plot.area(self, x=None, y=None, stacked=True, **kwargs)
# stack=False makes the graphs overlap. By default, alpha=0.5 in such a case.
df.plot.area(x='day', y=['fish', 'bread'])

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

import_path = r'.\pandas\drinks.csv'
# Necessary in this case since 'NA'='North America' in this dataset.
data = pd.read_csv(import_path, na_filter=False)
df = pd.DataFrame(data)
# Histogram.
# Data is divided into bins but are not categorical.
# Single variable.
# Good at displaying distribution (no gaps) of numerically finite data.
# Alternatively, use df.hist(). Hist displays the frequency of which individual
# numbers in a list of numbers occur. bin determines the number of blocks,
# range determines the total width of those blocks.
# by='col' specifies the column to group by, although it's not as often used.
df.plot(kind='hist', y=['beer_servings', 'spirit_servings'], bins=10,
        range=(0, 501), edgecolor='black', subplots=True, layout=(1, 2), sharey=True)

# df.hist can also be used to plot histograms. One key difference is the by
# argument, which allows for plotting a variable based on categories.
df.hist(column='beer_servings', by='continent', sharex=True, sharey=True, layout=(2, 3))

# Boxplot.
# Good for identifying outliers.
# The box represents the 25th, 50th and 75th percentile. The whiskers extend
# up to 1.5 x the interquartile (75th - 25th percentile) range, or the max or
# min value, whichever is reached first.
df.plot(kind='box')
