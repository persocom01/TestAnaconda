import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV

# Use this command if using Jupyter notebook to plot graphs inline.
# %matplotlib inline

data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df.columns = [x.lower().replace(' ', '_') for x in df.columns]
print(df.head())
print()

features = df.columns
target = pd.DataFrame(data.target)
X = df
y = target.values.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Using any of the special linear regressions requires the features to be
# scaled. You should also remove features that are highly correlated with each
# other whichw as not done here.
ss = StandardScaler()
X_train_ss = pd.DataFrame(ss.fit_transform(X_train[features]), columns=features)

# Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True,
# max_iter=None, tol=0.001, solver=’auto’, random_state=None)
# Present in all the models are the following attributes:
# .coef_ returns the coefficients for each feature.
# .intercept_ returns the y intercept.
# The ridge regression is the normal linear regression to use when using a LR
# for prediction. It is more resistant to multicollinearity than the default
# LR but it is susceptible to outliers.
# If you need a model that calculates alpha for itself, use RidgeCV().
ridge = Ridge()
# RidgeCV(alphas=(0.1, 1.0, 10.0), fit_intercept=True, normalize=False,
# scoring=None, cv=None, gcv_mode=None, store_cv_values=False)
# RidgeCV.alpha_ returns the regularization parameter.
ridge_cv = RidgeCV(alphas=np.linspace(.1, 10, 50), cv=5)
# LassoCV(eps=0.001, n_alphas=100, alphas=None, fit_intercept=True,
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
lasso = LassoCV(n_alphas=50, cv=5)
# ElasticNetCV(l1_ratio=0.5, eps=0.001, n_alphas=100, alphas=None,
# fit_intercept=True, normalize=False, precompute=’auto’, max_iter=1000,
# tol=0.0001, cv=’warn’, copy_X=True, verbose=0, n_jobs=None, positive=False,
# random_state=None, selection=’cyclic’)
# A combination of Ridge and Lasso for large datasets.
# l1_ratio determines how much of Ridge or Lasso to use. 1 is Lasso, 0 is
# Ridge.
elastic = ElasticNetCV(cv=5)

# Demonstrates searching for alpha for a model randomly.
# RandomizedSearchCV(estimator, param_distributions, n_iter=10, scoring=None,
# n_jobs=None, iid=’warn’, refit=True, cv=’warn’, verbose=0,
# pre_dispatch=‘2*n_jobs’, random_state=None, error_score=’raise-deprecating’,
# return_train_score=False)
# estimator=model
# param_distributions=dict where the keys=model_kwargs and values the values
# for those kwargs.
# n_iter=int determines the number of random values tested.
n_values = 50
random_search = RandomizedSearchCV(ridge, param_distributions={
                                   'alpha': (np.random.rand(n_values))*10}, n_iter=n_values, cv=5, random_state=1)

# Cross validation phase.
ridge_scores_rand = cross_val_score(random_search, X_train, y_train, cv=5)
print('ridge_random:', ridge_scores_rand.mean())
ridge_scores = cross_val_score(ridge, X_train, y_train, cv=5)
print('ridge:', ridge_scores.mean())
lasso_scores = cross_val_score(lasso, X_train, y_train, cv=5)
print('lasso:', lasso_scores.mean())
elastic_scores = cross_val_score(elastic, X_train, y_train, cv=5)
print('elastic:', elastic_scores.mean())

# Model fitting and evaluation.
ridge.fit(X_train, y_train)
y_hat = ridge.predict(X_test)

# Plot that visualizes the effect on the coefficients. Not that obvious unless
# you use lasso, which makes smaller coefficients zero.
pd.Series(ridge.coef_, index=features).plot.bar(figsize=(16, 10))
plt.title('coefficients plot')
plt.show()
plt.clf()

# A residual vs fitted values plot.
residuals = y_test - y_hat
plt.figure(figsize=(12, 7.5))
plt.scatter(y_hat, residuals)
plt.title('residual vs fitted values')
plt.axhline(y=0, color='k', lw=1)
plt.show()
plt.close()
