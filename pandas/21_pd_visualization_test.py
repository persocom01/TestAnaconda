# It is said that the most important visualization properties are
# position, color, and size.
import pandas as pd
import matplotlib.pyplot as plt

import_path = r'.\pandas\SacramentocrimeJanuary2006.csv'
data = pd.read_csv(import_path)
df = pd.DataFrame(data)
print(data)

# Bar chart.
# Categorical data separated by gaps between bars and larger gaps
# between categories.
# Multiple variables possible.
# Numerical data.
# Good for quick comparision.

# Pie chart.
# Categorical data.
# Single variable unless pie chart within pie charts. Can also be
# donut shaped.
# Percentage data.
# Good at showing relative proportions.

# Scatterplot.
# Made up of many points such as to appear continuous.
# Two variables only.
# Numerical data.
# Good at comparing two specific attributes of dataset(s) so as to
# displaying trends or relationships.

# Line graph.
# Single variable.
# Good at displaying the trend or the change over time of a
# variable alone or with other datasets.

# Histogram.
# Data is divided into bins but are not categorical.
# Single variable.
# Good at displaying distribution (no gaps) of numerically finite
# data.
