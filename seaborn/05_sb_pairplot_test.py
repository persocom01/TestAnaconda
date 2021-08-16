import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

import_path = r'./datasets/drinks.csv'
data = pd.read_csv(import_path)
df = pd.DataFrame(data)
print(df.head())

# sb.pairplot(data, hue=None, hue_order=None, palette=None, vars=None,
# x_vars=None, y_vars=None, kind='scatter', diag_kind='auto', markers=None,
# height=2.5, aspect=1, dropna=True, plot_kws=None, diag_kws=None,
# grid_kws=None, size=None)
# hue=col accepts a categorical variable which it uses to change the colors
# in the pairplot.
# vars and xy_vars accept lists of column names if you do not wish to pairplot
# all variables in the DataFrame.
# kind='reg' appears to plots a linear regression line for each pair.
# diag_kind can be 'hist' or 'kde' but it's mainly determined by whether the
# hue argument is set or not. It is hist by default, kdw when hue is given.
sb.pairplot(df, hue='continent', palette='husl')
plt.show()
plt.close()
