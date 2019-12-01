# Demonstrates ways to feature select for categorical variables.
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sb
from itertools import combinations

# Use this command if using Jupyter notebook to plot graphs inline.
# %matplotlib inline

# Data dictionary found here:
# https://www.kaggle.com/uciml/mushroom-classification
import_path = r'.\datasets\mushrooms.csv'
data = pd.read_csv(import_path)

df = pd.DataFrame(data)
# This column contains only one value and is thus useless.
del df['veil-type']


def cramers_corr(df):
    '''
    Takes a DataFrame of categorical variables and returns a DataFrame of the
    correlation matrix based on the Cramers V statistic. Uses correction from
    Bergsma and Wicher, Journal of the Korean Statistical Society 42 (2013):
    323-328

    Does not require variables to be label encoded before use.
    '''

    def cramers_v(x, y):
        con_matrix = pd.crosstab(x, y)
        chi2 = stats.chi2_contingency(con_matrix)[0]
        n = con_matrix.sum().sum()
        phi2 = chi2/n
        r, k = con_matrix.shape
        phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
        rcorr = r-((r-1)**2)/(n-1)
        kcorr = k-((k-1)**2)/(n-1)
        return np.sqrt(phi2corr/min((kcorr-1), (rcorr-1)))

    cols = df.columns
    n_cols = len(cols)
    corr_matrix = np.zeros((n_cols, n_cols))
    for col1, col2 in combinations(cols, 2):
        i1, i2 = cols.get_loc(col1), cols.get_loc(col2)
        corr_matrix[i1, i2] = cramers_v(df[col1], df[col2])
        corr_matrix[i2, i1] = corr_matrix[i1, i2]

    df_corr_matrix = pd.DataFrame(corr_matrix, index=cols, columns=cols)

    return df_corr_matrix


fig, ax = plt.subplots(figsize=(12, 7.5))
ax = sb.heatmap(cramers_corr(df), annot=True, ax=ax)
ax.set_title("Cramers V Correlation between Variables")
