# Stochastic Gradient Descent (SGD) is an algorithm used to solve linear
# systems. Stochastic is really just a fancy word for random. Linear systems
# can be solved directly or iteratively, and SGD falls into the latter
# category. The main reason to use SGD is on problems may be too
# computationally complex to reasonably solve using other methods. The more
# variables and the bigger the amount of data, the higher the complexity. Even
# so, from experience, SGD does not solve nlp problems with thousands of
# features in reasonable time.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import explained_variance_score
import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import pleiades as ple

data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df.columns = [x.lower().replace(' ', '_') for x in df.columns]
target = 'target'
features = [col for col in df.columns if col != target]
df[target] = data.target
print(df.head())
print()

X = df[features]
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

sebas = ple.Sebastian()

pipe = Pipeline([
    ('ss', StandardScaler()),
    ('sgd', SGDRegressor())
])
params = {
    'sgd__alpha': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
}
gs = GridSearchCV(pipe, param_grid=params, cv=5, n_jobs=-1)
gs.fit(X_train, y_train)

y_pred = gs.predict(X_test)

print('mean absolute error(MAE):', round(
    mean_absolute_error(y_test, y_pred), 2))
# For MSE square=True by default.
print('root mean squared error(RMSE):', round(
    mean_squared_error(y_test, y_pred, squared=False), 2))
print('median absolute error:', round(
    median_absolute_error(y_test, y_pred), 2))
# explained_variance_score(y_true, y_pred, sample_weight=None,
# multioutput=’uniform_average’)
# explained_variance_score is in effect an unbiased version of R**2. R**2 will
# return the same result regardless of the bias of the model, reflecting only
# the closeness of the slope of the prediction with the slope test values,
# regardless of the y intercept (bias).
print('explained variance score:', round(
    explained_variance_score(y_test, y_pred), 2))
# r2_score(y_true, y_pred, sample_weight=None,
# multioutput=’uniform_average) returns the R**2 value of the prediction,
# where:
# r_square = 1 - (sum of squared residuals) / (total sum of squares)
# squared residuals being (actual value - prediction)**2,
# and squares being (actual value - mean)**2.
# The maximum value of r2 is 1.0, but can be negative, if the model performs
# worse than the mean.
print('r_square:', round(r2_score(y_test, y_pred), 2))
print()


# Adj r2 = 1-(1-R2)*(n-1)/(n-p-1) where n=sample_size and p=number_of_x_vars.
def adj_r2(X, y, y_pred):
    return 1 - (1-r2_score(y, y_pred))*(len(y)-1)/(len(y)-X.shape[1]-1)


# Plot the single x variable linear regression graph.
# fig, ax = plt.subplots(figsize=(12, 7.5))
# ax.plot([], [], ' ', label=r'$R^2 = $' +
#         f'{round(adj_r2(X_test, y_test, y_pred), 2)}')
# ax.scatter(X_test, y_test, alpha=0.7, label='test set y values')
# ax.plot(X_test, y_pred, color='g', alpha=0.7, label='linear regression line')
# ax.legend()
# for i, y_h in enumerate(y_pred):
#     ax.plot([X_test[i], X_test[i]], [y_h, y_test[i]], color='r', alpha=0.7)
# plt.show()
# plt.close()

# Plot the predicted vs actual y graph for multiple x value linear regression.
fig, ax = plt.subplots(figsize=(12, 7.5))
ax.plot([], [], ' ', label=r'adj $R^2 = $' +
        f'{round(adj_r2(X_test, y_test, y_pred), 2)}')
ax.scatter(y_pred, y_test, alpha=0.7, label='tested y values')
ax.plot(y_pred, y_pred, color='g', alpha=0.7, label='predicted y values')
ax.legend()
for i, y_h in enumerate(y_pred):
    ax.plot([y_pred[i], y_pred[i]], [y_h, y_test[i]], color='r', alpha=0.7)
plt.show()
plt.close()

# Demonstrates a residual vs fitted values plot.
residuals = y_test - y_pred
plt.figure(figsize=(12, 7.5))
plt.scatter(y_pred, residuals)
plt.title('residual vs fitted values')
plt.axhline(y=0, color='k', lw=1)
plt.show()
plt.close()
