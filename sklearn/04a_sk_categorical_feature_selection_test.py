# Demonstrates ways to feature select for categorical variables.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import pleiades as ple

# Use this command if using Jupyter notebook to plot graphs inline.
# %matplotlib inline

# Data dictionary found here:
# https://www.kaggle.com/uciml/mushroom-classification
import_path = r'./datasets/mushrooms.csv'
data = pd.read_csv(import_path, index_col=None)

df = pd.DataFrame(data)
# This column contains only one value and is thus useless.
del df['veil-type']

sol = ple.Solution()

fig, ax = plt.subplots(figsize=(16, 10))
ax = sb.heatmap(sol.cramers_corr(df), annot=True, ax=ax, cmap='Greens')
ax.set_title('Cramers V Correlation between Categorical Variables')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom+0.5, top-0.5)
plt.show()
