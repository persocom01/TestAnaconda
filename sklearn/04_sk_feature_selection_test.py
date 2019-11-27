import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE

# Use this command if using Jupyter notebook to plot graphs inline.
# %matplotlib inline

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df.columns = [x.lower().replace(' ', '_') for x in df.columns]
target = 'cancer'
features = [col for col in df.columns if col != target]
df[target] = data.target
print(df.head())
print()

# One of the easiest ways to eliminate features is through a heatmap.
# A correlation of 0.2 and below is considered low. 0.75 and above is
# considered high. Features with high correlations between themselves need to
# be eliminated using VIF.
fig, ax = plt.subplots(figsize=(12, 7.5))
sb.heatmap(df.corr(), cmap='PiYG', annot=False)
# Corrects the heatmap for later versions of matplotlib.
bottom, top = ax.get_ylim()
ax.set_ylim(bottom+0.5, top-0.5)
# plt.show()

X = df[features]
y = df[target].values

# VIF, or Variance Inflation Factor, is a measure of colinearity among
# predictor variables within a multiple regression. It is used to eliminate
# features that are highly correlated with each other.
# Variables with the highest VIF scores should be eliminated until the VIF
# scores are below between 10 to 2.5, depending on how conservative you want
# to be. This function either takes a VIF score to eliminate features until,
# or the number of features you want returned.
# It seems to give funny results if the variables contain string values.
# Recommend separating the string columns from the rest before attempting
# to do this.


def vif_feature_select(df, max_score=5.0, n_features=-1, inplace=False, drop_list=False, drops=None):
    '''
    Takes a DataFrame and returns it after recursively eliminating columns
    with the highest VIF scores until either the remainder have a VIF score
    of less than max_score, or there are n_features left.
    '''
    if not inplace:
        df = df.copy()
    if not drops:
        drops = []
    features = df.columns
    vifs = np.linalg.inv(df.corr().values).diagonal()
    max_vif_index = np.argmax(vifs)
    if n_features < 0 and vifs[max_vif_index] >= max_score:
        drops.append(features[max_vif_index])
        del df[features[max_vif_index]]
        return vif_feature_select(df, max_score, n_features, inplace, drop_list, drops)
    elif n_features >= 0 and len(features) > n_features:
        drops.append(features[max_vif_index])
        del df[features[max_vif_index]]
        return vif_feature_select(df, max_score, n_features, inplace, drop_list, drops)
    else:
        if drop_list:
            print('returning list of dropped features.')
            return drops
        else:
            return df


print('VIF:')
X = vif_feature_select(X)
features = X.columns
print(features)
print()

fig, ax = plt.subplots(figsize=(12, 7.5))
sb.heatmap(X.corr(), cmap='PiYG', annot=False)
# Corrects the heatmap for later versions of matplotlib.
bottom, top = ax.get_ylim()
ax.set_ylim(bottom+0.5, top-0.5)
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=data.target)

# SelectKBest(score_func=<function f_classif>, k=10)
# k sets the ending number of desired features.
# chi2(X, y) is the chi2 test used to compare a categorical y with non
# zero features x.
# Use skb.scores_ to see the actual chi2 score.
skb = SelectKBest(score_func=chi2, k=5)
skb.fit(X_train, y_train)
# Preserves the column names compared to a straight skb.fit_transform()
selected_cols = [v for i, v in enumerate(
    X_train.columns) if i in skb.get_support(indices=True)]
X_train_chi2 = X_train[selected_cols]
ranking = [{v: k for k, v in enumerate(
    np.sort(skb.scores_)[::-1])}.get(r) for r in skb.scores_]
print('chi2 feature selection:')
print('feature ranking (0 being best):', ranking)
print(X_train_chi2.columns)
print()

# RFE(estimator, n_features_to_select=None, step=1, verbose=0) performs
# recursive feature elimination of features.
# step determines the number of features to remove at each iteration. If step
# is between 0.0 and 1.0, it is taken as the proportion of total features.
lm = LinearRegression()
rfe = RFE(lm, 5)
rfe.fit(X_train, y_train)
selected_cols = [v for i, v in enumerate(
    X_train.columns) if i in rfe.get_support(indices=True)]
X_train_rfe = X_train[selected_cols]
print('recursive feature elimination:')
print('feature ranking (1 being best):', rfe.ranking_)
print(X_train_rfe.columns)
print()

# PCA(n_components=None, copy=True, whiten=False, svd_solver=’auto’,
# tol=0.0, iterated_power=’auto’, random_state=None)
# Principal component analysis.
# n_components determines number of features to keep.
ss = StandardScaler()
X_train_ss = pd.DataFrame(ss.fit_transform(X_train[features]), columns=features)
n_comp = 5
pca = PCA(n_components=n_comp)
pca.fit(X_train_ss)
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_comp)]
most_important_names = [features[most_important[i]] for i in range(n_comp)]
X_train_pca = X_train[most_important_names]
# How much of the variance is explained by each feature.
print('Principal component analysis:')
print(pca.explained_variance_ratio_)
print(X_train_pca.columns)
print()
