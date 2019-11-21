import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as skds
import sklearn.feature_selection as skfs
import sklearn.model_selection as skms
import sklearn.linear_model as sklm
import sklearn.metrics as skm

# Use this command if using Jupyter notebook to plot graphs inline.
# %matplotlib inline

data = skds.load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
print(df.head())
print()

features = data.feature_names
X = df[features]
y = data.target

X_train, X_test, y_train, y_test = skms.train_test_split(X, y, random_state=1, stratify=data.target)

# skfs.SelectKBest(score_func=<function f_classif>, k=10)
# k sets the ending number of desired features.
# skfs.chi2(X, y) is the chi2 test used to compare a categorical y with non
# zero features x.
print('chi2 selected columns:')
# Use skb.scores_ to see the actual chi2 score.
skb = skfs.SelectKBest(score_func=skfs.chi2, k=10)
skb.fit(X_train, y_train)
# Preserves the column names compared to a straight skb.fit_transform()
selected_cols = [v for i, v in enumerate(X_train.columns) if i in skb.get_support(indices=True)]
X_train_chi2 = X_train[selected_cols]
print(X_train_chi2.columns)
print()

# skfs.RFE(estimator, n_features_to_select=None, step=1, verbose=0) performs
# recursive feature elimination
lm = sklm.LinearRegression()
rfe = skfs.RFE(lm, 3)
