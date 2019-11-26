import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as skds
import sklearn.preprocessing as skpp
import sklearn.model_selection as skms
import sklearn.linear_model as sklm
import sklearn.metrics as skm

# Use this command if using Jupyter notebook to plot graphs inline.
# %matplotlib inline

data = skds.load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df.columns = [x.lower().replace(' ', '_') for x in df.columns]
print(df.head())
print()

features = df.columns
target = pd.DataFrame(data.target)
X = df
y = target.values.ravel()

X_train, X_test, y_train, y_test = skms.train_test_split(X, y, random_state=1)

# Using any of the special linear regressions requires the features to be
# scaled.
ss = skpp.StandardScaler()
X_train_ss = pd.DataFrame(ss.fit_transform(X_train[features]), columns=features)

# sklm.RidgeCV(alphas=(0.1, 1.0, 10.0), fit_intercept=True, normalize=False,
# scoring=None, cv=None, gcv_mode=None, store_cv_values=False)
# sklm.RidgeCV.coef_ returns the coefficients for each feature.
# sklm.RidgeCV.intercept_ returns the y intercept.
# sklm.RidgeCV.alpha_ returns the regularization parameter.
# The ridge regression is the normal linear regression to use when using a LR
# for prediction. It is more resistant to multicollinearity than the default
# LR but it is susceptible to outliers.
ridge = sklm.RidgeCV(alphas=np.linspace(.1, 10, 100), cv=5)
# sklm.LassoCV(eps=0.001, n_alphas=100, alphas=None, fit_intercept=True,
# normalize=False, precompute=’auto’, max_iter=1000, tol=0.0001, copy_X=True,
# cv=’warn’, verbose=False, n_jobs=None, positive=False, random_state=None,
# selection=’cyclic’)
# n_jobs=-1 allows your computer to use all its cores for the computation.
# Lasso is more computationally intensive than ridge for a small number of
# features.
# The lasso regression is more resistant to outliers compared to ridge, and it
# has the added advantage of being able to eliminate less important features.
# For this reason it is used when the number of features is very large and you
# don't know which are important.
lasso = sklm.LassoCV(n_alphas=200, cv=5)
# ElasticNetCV(l1_ratio=0.5, eps=0.001, n_alphas=100, alphas=None,
# fit_intercept=True, normalize=False, precompute=’auto’, max_iter=1000,
# tol=0.0001, cv=’warn’, copy_X=True, verbose=0, n_jobs=None, positive=False,
# random_state=None, selection=’cyclic’)
# A combination of Ridge and Lasso for large datasets.
# l1_ratio determines how much of Ridge or Lasso to use. 1 is Lasso, 0 is
# Ridge.
elastic = sklm.ElasticNetCV(cv=5)

# Cross validation phase.
ridge_scores = skms.cross_val_score(ridge, X_train, y_train, cv=5)
print('ridge:', ridge_scores.mean())
lasso_scores = skms.cross_val_score(lasso, X_train, y_train, cv=5)
print('lasso:', lasso_scores.mean())
elastic_scores = skms.cross_val_score(elastic, X_train, y_train, cv=5)
print('elastic:', elastic_scores.mean())

# Model fitting and evaluation.
ridge.fit(X_train, y_train)
y_hat = ridge.predict(X_test)

# Plot that visualizes the effect on the coefficients. Not that obvious unless
# you use lasso, which makes smaller coefficients zero.
pd.Series(ridge.coef_, index=features).plot.bar(figsize=(16, 10))
plt.show()
plt.clf()

# Demonstrates a residuals plot. It is used to diagnose 3 possible flaws
# in a linear regression:
# 1. The errors have a visible pattern.
# This indicates that the linear model did not adequately capture the variation
# in the target, and the need to use polynomial terms.
# 2. The errors have a funnel shape.
# This indicates the presence of heteroskedasticity, which means the error
# variance errors is not constant. This may be an indication that outliers
# are having a large impact on the distribution, which may in turn cause the
# confidence intervals of the prediction to be too wide or too narrow.
# 3. The errors are not normally distributed with mean 0.
# This can cause confidence intervals of the prediction to be too wide or too
# narrow. This may be an indication of unusual data points that need further
# study.
residuals = y_test - y_hat
plt.figure(figsize=(16, 10))
plt.scatter(y_hat, residuals)
plt.title("residual errors")
plt.axhline(y=0, color='k', lw=1)
plt.show()
plt.close()
