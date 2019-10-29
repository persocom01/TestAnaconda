# It is said that the most important visualization properties are
# position, color, and size.
import pandas as pd
import matplotlib.pyplot as plt

data = {'day': [1, 2, 3, 4, 5, 6], 'temperature': [43, 53, 50, 57, 59, 67]}
df = pd.DataFrame(data)
# Line graph.
# Single variable.
# Good at displaying the trend or the change over time of a
# variable alone or with other datasets.
# df.plot(kind) is the basic pandas vistualization method.
# It will plot a line graph by default, but the argument
# kind='graph_name' can be used specify other graphs:
# 'bar' = bar graph
# 'hist' = histogram
# ‘box’ for boxplot
# ‘kde’ or ‘density’ for density plots
# ‘area’ for area plots
# ‘scatter’ for scatter plots
# ‘hexbin’ for hexagonal bin plots
# ‘pie’ for pie plots
df.plot(x='day', y='temperature')

data = {'movie': ['comedy', 'action', 'romance', 'drama', 'scifi'], 'number': [4, 5, 6, 1, 4]}
df = pd.DataFrame(data)
# Bar chart.
# Categorical data separated by gaps between bars and larger gaps
# between categories.
# Multiple variables possible.
# Numerical data.
# Good for quick comparision.
df.plot(kind='bar', x='movie')

# Pie chart.
# Categorical data.
# Single variable unless pie chart within pie charts. Can also be
# donut shaped.
# Percentage or proportional data.
# Good at showing relative proportions.
df.plot(kind='pie', y='number', labels=df['movie'])

import_path = r'.\pandas\SacramentocrimeJanuary2006.csv'
data = pd.read_csv(import_path)
df = pd.DataFrame(data)
# Scatterplot.
# Made up of many points such as to appear continuous.
# Two variables only.
# Numerical data.
# Good at comparing two specific attributes of dataset(s) so as to
# displaying trends or relationships.
df.plot(kind='scatter', x='longitude', y='latitude')

# Histogram.
# Data is divided into bins but are not categorical.
# Single variable.
# Good at displaying distribution (no gaps) of numerically finite
# data.
