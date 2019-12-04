# Logistic regression is a model used for binary or categorical target
# predictive problems. It differs from KNN in that it is strictly a linear
# model, and thus may require polynomial feature creation to work correctly.
# However, it can return probabilities instead of labels, which can be useful
# in some cases.
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
import data_plots as dp

# Use this command if using Jupyter notebook to plot graphs inline.
# %matplotlib inline

data = load_breast_cancer()
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

# Logistic regression itself doesn't require scaling, however, because the
# function is regularized by default, and regularization requires scaling,
# scale before use.
ss = StandardScaler()
X_train = pd.DataFrame(ss.fit_transform(X_train), columns=features)
X_test = pd.DataFrame(ss.transform(X_test), columns=features)

# LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0,
# fit_intercept=True, intercept_scaling=1, class_weight=None,
# random_state=None, solver='warn', max_iter=100, multi_class='warn',
# verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
lr = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000)
lr.fit(X_train, y_train)

# cross_val_predict(estimator, X, y=None, groups=None, cv='warn', n_jobs=None,
# verbose=0, fit_params=None, pre_dispatch='2*n_jobs', method='predict')
# Unlike cross_val_score, cross_val_predict doesn't split the data into folds
# and return the score for each fold. Instead, it returns the y_pred for each
# X_train. The number of folds only reduces the size of the dataset used to
# train the model. At cv=5, 1/5 of the dataset is withheld during model
# fitting. Used with an evaluation metric, you can test the model for accuracy.
cv = cross_val_predict(lr, X_train, y_train, cv=5)
print('cross val accuracy:', (accuracy_score(cv, y_train)*100))
print()

y_pred = lr.predict(X_test)
# You use predict_proba when you wish to use a different threshold probability
# for determining if the target is 0 or 1. It is possible to map a new list of
# predictions based on this new probability.
y_pred_prob = lr.predict_proba(X_test)
print('logreg predict vs y_test:')
print(y_pred)
print(y_test)
print()
print('logreg prob:')
print(y_pred_prob[:5])
print()

print('logreg accuracy:')
print((accuracy_score(y_test, y_pred)*100))
print()

# Logistic regression is interpreted by as a unit increase in x makes the
# odds of Chronic Kidney Disease e^coefficient more likely. To get a more
# interpretable number, use np.exp.
print('coefficient as odds:')
print(np.exp(lr.coef_))
print()

# Plots the roc curve.
roc = dp.Roc(y_test, y_pred)
roc.plot()
