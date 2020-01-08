import numpy as np
import pandas as pd
import pleiades as ple
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

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
# fig, ax = plt.subplots(figsize=(12, 7.5))
# sb.heatmap(df.corr(), cmap='PiYG', annot=False)
# # Corrects the heatmap for later versions of matplotlib.
# bottom, top = ax.get_ylim()
# ax.set_ylim(bottom+0.5, top-0.5)
# plt.show()
# plt.close()

X = df[features]
y = df[target].values

sol = ple.Solution()

# VIF, or Variance Inflation Factor, is a measure of colinearity among
# predictor variables within a multiple regression. It is used to eliminate
# continuous or ordinal features that are highly correlated with each other.
# Variables with the highest VIF scores should be eliminated until the VIF
# scores are below between 10 to 2.5, depending on how conservative you want
# to be. This function either takes a VIF score to eliminate features until,
# or the number of features you want returned.
# It seems to give funny results if the variables contain string values.
# Recommend separating the string columns from the rest before attempting
# to do this.
print('VIF:')
X = sol.vif_feature_select(X)
features = X.columns
print(features)
print()

# fig, ax = plt.subplots(figsize=(12, 7.5))
# sb.heatmap(X.corr(), cmap='PiYG', annot=False)
# # Corrects the heatmap for later versions of matplotlib.
# bottom, top = ax.get_ylim()
# ax.set_ylim(bottom+0.5, top-0.5)
# plt.show()
# plt.close()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=1, stratify=data.target)

# A rule of thub as to an acceptable number of features is
# n_features = sqrt(n_rows)
n_features = 5

# SelectKBest(score_func=<function f_classif>, k=10)
# k sets the ending number of desired features.
# chi2(X, y) is the chi2 test used to compare a categorical y with non
# zero features x.
# Use skb.scores_ to see the actual chi2 score.
skb = SelectKBest(score_func=chi2, k=n_features)
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
rfe = RFE(lm, n_features)
rfe.fit(X_train, y_train)
selected_cols = [v for i, v in enumerate(
    X_train.columns) if i in rfe.get_support(indices=True)]
X_train_rfe = X_train[selected_cols]
print('recursive feature elimination:')
print('feature ranking (1 being best):', rfe.ranking_)
print(X_train_rfe.columns)
print()

# PCA(n_components=None, copy=True, whiten=False, svd_solver='auto',
# tol=0.0, iterated_power='auto', random_state=None)
# Principal component analysis.
# Can be used for feature selection, especially when the goal of the subsequent
# analysis is to find clusters as it is a good at removing noise. When used to
# transform features, it removes multicollinearity. However, it should be noted
# PCA assumes that features have linear relationships.
# n_components=0-1_int determines number of features to keep. By default, all
# components are kept. If set to a number between 0 and 1 and
# svd_solver='full', components are kept until explained_variance_ is greater
# than the number.
# whiten=True sets the variance for each vector to 1.
# svd_solver='auto' == svd_solver='full' as long as the matrix is smaller than
# 500x500. It might be 'arpack' otherwise.
ss = StandardScaler()
X_train = pd.DataFrame(ss.fit_transform(X_train), columns=features)
X_test = pd.DataFrame(ss.transform(X_test), columns=features)
pca = PCA(n_components=n_features)
pca.fit(X_train)
most_important = [np.abs(pca.components_[i]).argmax()
                  for i in range(n_features)]
most_important_names = [features[most_important[i]] for i in range(n_features)]
pca_features = {k: v for k, v in zip(
    most_important_names, pca.explained_variance_ratio_)}
print('Principal component analysis:')
# pca transform eliminates the correlations between the features, but the
# DataFrame will now be in terms of PCs.
print(pd.DataFrame(pca.transform(X_train), columns=[
      'PC' + str(i+1) for i in range(n_features)]).head())

# Visualizing the features that contribute to the PCs can be difficult.


def plot_pc(pc, features=None):
    if features is None:
        features = range(len(pc))
    fig, ax = plt.subplots()
    pc = pc[::-1]
    features = features[::-1]
    ax.barh(features, pc)


def plot_two_pcs(pc1, pc2):
    fig, ax = plt.subplots()
    for p0, p1 in zip(pca.components_[0], pca.components_[1]):
        plt.plot([0, p0], [0, p1])


plot_pc(pca.components_[0], features)
plt.show()
plt.close()
# How much of the variance is explained by each feature.
print(pca_features)
# Restores the original order.
most_important_names = [
    x for x in X_train.columns if x in most_important_names]
X_train_pca = X_train[most_important_names]
print(X_train_pca.columns)
print()

# Plot cumulative variance explains. There is no hard and fast rule as to how
# much of the variance has to be explained to be considered enough, but roughly
# 80-90% is probably okay.
fig, ax = plt.subplots(figsize=(12, 7.5))
ax.plot(range(n_features), np.round(pca.explained_variance_ratio_,
                                    decimals=4), label='Variance explained')
cve = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4))
ax.plot(range(n_features), cve, label='Cumulative variance explained')
# Reduces number of x ticks in order to eliminate the xtick decimal place.
plt.locator_params(nbins=len(cve))
ax.legend()
ax.set_ylim(0, 1)
ax.set_title('Variance explained vs Components')
ax.set_xlabel('Principal components')
ax.set_ylabel('Variance explained')
plt.show()
plt.close()

# TruncatedSVD(n_components=2, algorithm='randomized', n_iter=5,
# random_state=None, tol=0.0)
# This is PCA without centering the data. Centering makes the mean the
# reference point for the y intercept for something. Otherwise, there should be
# no difference. The main reason we may prefer to use TruncatedSVD, however, is
# that it works on sparse matrices produced by NLP word vectorizers when PCA
# may not.
tsvd = TruncatedSVD(n_components=n_features)
tsvd.fit(X_train)
most_important = [np.abs(tsvd.components_[i]).argmax()
                  for i in range(n_features)]
most_important_names = [features[most_important[i]] for i in range(n_features)]
tsvd_features = {k: v for k, v in zip(
    most_important_names, tsvd.explained_variance_ratio_)}
print('TruncatedSVD:')
# How much of the variance is explained by each feature.
print(tsvd_features)
# Restores the original order.
most_important_names = [
    x for x in X_train.columns if x in most_important_names]
X_train_tsvd = X_train[most_important_names]
print(X_train_tsvd.columns)
print()
