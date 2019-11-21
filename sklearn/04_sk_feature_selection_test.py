import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn.datasets as skds
import sklearn.preprocessing as skpp
import sklearn.feature_selection as skfs
import sklearn.decomposition as skd
import sklearn.model_selection as skms
import sklearn.linear_model as sklm

# Use this command if using Jupyter notebook to plot graphs inline.
# %matplotlib inline

data = skds.load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['cancer'] = data.target
print(df.head())
print()

# One of the easiest ways to eliminate features is through a heatmap.
fig, ax = plt.subplots(figsize=(12, 7.5))
sb.heatmap(df.corr(), cmap='PiYG', annot=False)
# Corrects the heatmap for later versions of matplotlib.
bottom, top = ax.get_ylim()
ax.set_ylim(bottom+0.5, top-0.5)
# plt.show()

features = data.feature_names
X = df[features]
y = data.target

X_train, X_test, y_train, y_test = skms.train_test_split(X, y, random_state=1, stratify=data.target)

# skfs.SelectKBest(score_func=<function f_classif>, k=10)
# k sets the ending number of desired features.
# skfs.chi2(X, y) is the chi2 test used to compare a categorical y with non
# zero features x.
print('chi2 feature selection:')
# Use skb.scores_ to see the actual chi2 score.
skb = skfs.SelectKBest(score_func=skfs.chi2, k=5)
skb.fit(X_train, y_train)
# Preserves the column names compared to a straight skb.fit_transform()
selected_cols = [v for i, v in enumerate(X_train.columns) if i in skb.get_support(indices=True)]
X_train_chi2 = X_train[selected_cols]
ranking = [{v: k for k, v in enumerate(np.sort(skb.scores_)[::-1])}.get(r) for r in skb.scores_]
print('feature ranking (0 being best):', ranking)
print(X_train_chi2.columns)
print()

# skfs.RFE(estimator, n_features_to_select=None, step=1, verbose=0) performs
# recursive feature elimination of features.
# step determines the number of features to remove at each iteration. If step
# is between 0.0 and 1.0, it is taken as the proportion of total features.
print('recursive feature elimination:')
lm = sklm.LinearRegression()
rfe = skfs.RFE(lm, 5)
rfe.fit(X_train, y_train)
selected_cols = [v for i, v in enumerate(X_train.columns) if i in rfe.get_support(indices=True)]
X_train_rfe = X_train[selected_cols]
print('feature ranking (1 being best):', rfe.ranking_)
print(X_train_rfe.columns)
print()

# skd.PCA(n_components=None, copy=True, whiten=False, svd_solver=’auto’,
# tol=0.0, iterated_power=’auto’, random_state=None)
# n_components determines number of features to keep.
ss = skpp.StandardScaler()
X_train_ss = pd.DataFrame(ss.fit_transform(X_train[features]), columns=features)
n_comp = 5
pca = skd.PCA(n_components=n_comp)
pca.fit(X_train_ss)
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_comp)]
most_important_names = [features[most_important[i]] for i in range(n_comp)]
dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_comp)}
X_train_pca = pd.DataFrame(dic.items())
# How much of the variance is explained by each feature.
print(pca.explained_variance_ratio_)
print(X_train_pca)
