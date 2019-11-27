import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import explained_variance_score

# Use this command if using Jupyter notebook to plot graphs inline.
# %matplotlib inline

data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
# Doesn't do anything in this case but common to simplify column names.
df.columns = [x.lower().replace(' ', '_') for x in df.columns]
print(df.head())
print()

# Typically, if the DataFrame is not already split into features and target,
# you can get a list of feature names with the following code, before passing
# it to the DataFrame as a list of columns:
# features = [col for col in df.columns if col != 'target']
target = pd.DataFrame(data.target)
# We use [:, np.newaxis] in this case to add a y axis to the output to make it
# a 2d array so it can be accepted by the LinearRegression() later.
# X = df['bmi'][:, np.newaxis]
# Multiple x values.
X = df
# Use df.values or np.array() to avoid problems later.
y = target.values

# train_test_split(arr_features, arr_target, test_size=0.25, **options)
# options accepts a number of arguments, including:
# test_size=float_int if given an int, it takes it as the absolute number of
# rows to take as the test data. If given a float, takes it as the proportion
# of the total data to take as the test set.
# random_state=int is the rng seed.
# shuffle=True randomizes the data before splitting.
# stratify=list lets you control the proportion of values in the split by
# specifying a categorical value for each row in the array. The split will
# have the same proportion of these categorical values. In practice you just
# pass it one of the DataFrame's categorial columns.
# In this case, the 'age' column was effectively made a categorical column
# where the proportion of 'age' below 0 is the same in the training and
# testing split.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=1, stratify=(df['age'] < 0))
print('train:')
print(X_train[:5])
print('count:', len(X_train))
print()
print('test:')
print(X_test[:5])
print('count:', len(X_test))
print()

# cross_val_score(estimator, X, y=None, groups=None, scoring=None,cv=’warn’,
# n_jobs=None, verbose=0, fit_params=None, pre_dispatch=‘2*n_jobs’,
# error_score=’raise-deprecating’) returns a list of length=cv of the R**2
# scores when the estimator is applied with X features to predict the target y.
# It is the first line of defense when choosing or rejecting a model with
# X features. If the variation in R**2 is bad, perhaps above 0.5 in difference,
# either the model or the feature composition of X may need to be changed.
# cv=int determines the number of sections to divide the data into. For cv=5,
# the data is divided into 5 equal parts, and a model train on 4 parts is
# tested on the last part.
print('cross_val_score:')
print(cross_val_score(LinearRegression(), X_train, y_train, cv=5))
print()

# Demonstrates training a model on test data.
# LinearRegression(fit_intercept=True, normalize=False, copy_X=True,
# n_jobs=None)
# LinearRegression() needs to be called again for every new model.
# normalize=True will deduct the mean from each value and divide it by the
# l2 norm, which is the root of the sum of the squares of all values.
# lr.fit(self, X, y, sample_weight=None)
lm = LinearRegression()
lm.fit(X_train, y_train)

# lm.predict(self, X) returns y values predicted by the model for input values
# of X. We normally pass it the X_test values because we want to see how close
# predicted y is to y_test.
y_hat = lm.predict(X_test)
print('predictions:')
print(y_hat[:10])
print('count:', len(y_hat))
print()

print('coefficients:', lm.coef_)
print()

print('intercept:', lm.intercept_)
print()

# Model performance measures.
print('mean absolute error(MAE):', round(
    mean_absolute_error(y_test, y_hat), 2))
print('mean squared error(MSE):', round(
    mean_squared_error(y_test, y_hat), 2))
print('median absolute error:', round(
    median_absolute_error(y_test, y_hat), 2))
# explained_variance_score(y_true, y_pred, sample_weight=None,
# multioutput=’uniform_average’)
# explained_variance_score is in effect an unbiased version of R**2. R**2 will
# return the same result regardless of the bias of the model, reflecting only
# the closeness of the slope of the prediction with the slope test values,
# regardless of the y intercept (bias).
print('explained variance score:', round(explained_variance_score(y_test, y_hat), 2))
# r2_score(y_true, y_pred, sample_weight=None,
# multioutput=’uniform_average) returns the R**2 value of the prediction,
# where:
# R**2 = 1 - (sum of squared residuals) / (total sum of squares)
# squared residuals being (actual value - prediction)**2,
# and squares being (actual value - mean)**2.
# The maximum value of R**2 is 1.0, but can be negative, if the model performs
# worse than the mean.
print('R**2:', round(r2_score(y_test, y_hat), 2))
print()


# Adj r2 = 1-(1-R2)*(n-1)/(n-p-1) where n=sample_size and p=number_of_x_vars.
def adj_r2(X, y, y_hat):
    return 1 - (1-r2_score(y, y_hat))*(len(y)-1)/(len(y)-X.shape[1]-1)


# Plot the single x variable linear regression graph.
# fig, ax = plt.subplots(figsize=(12, 7.5))
# ax.plot([], [], ' ', label=r'$R^2 = $' +
#         f'{round(adj_r2(X_test, y_test, y_hat), 2)}')
# ax.scatter(X_test, y_test, alpha=0.7, label='test set y values')
# ax.plot(X_test, y_hat, color='g', alpha=0.7, label='linear regression line')
# ax.legend()
# for i, y_h in enumerate(y_hat):
#     ax.plot([X_test[i], X_test[i]], [y_h, y_test[i]], color='r', alpha=0.7)
# plt.show()
# plt.clf()

# Plot the predicted vs actual y graph for multiple x value linear regression.
fig, ax = plt.subplots(figsize=(12, 7.5))
ax.plot([], [], ' ', label=r'adj $R^2 = $' +
        f'{round(adj_r2(X_test, y_test, y_hat), 2)}')
ax.scatter(y_hat, y_test, alpha=0.7, label='tested y values')
ax.plot(y_hat, y_hat, color='g', alpha=0.7, label='predicted y values')
ax.legend()
for i, y_h in enumerate(y_hat):
    ax.plot([y_hat[i], y_hat[i]], [y_h, y_test[i]], color='r', alpha=0.7)
plt.show()
plt.clf()

# Demonstrates a residual vs fitted values plot. It is used to diagnose 3
# possible flaws in a linear regression:
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
plt.figure(figsize=(12, 7.5))
plt.scatter(y_hat, residuals)
plt.title('residual vs fitted values')
plt.axhline(y=0, color='k', lw=1)
plt.show()
plt.close()
