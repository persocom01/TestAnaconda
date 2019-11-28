# Demonstrates the k nearest neighbors algorithm which is mainly used for
# categorical target predictive problems. It can be used on a continuous target
# as well, but there are probably better models for that. I works well on
# datasets with fewer features. It is unknown if it works on multiple targets.
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Use this command if using Jupyter notebook to plot graphs inline.
# %matplotlib inline

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df.columns = [x.lower().replace(' ', '_') for x in df.columns]
target = 'species'
features = [col for col in df.columns if col != target]
df[target] = data.target
print(df.head())
print()

# Categorical targets are great with seaborn pairplots since the hue of the
# plot can be set by the target.
# sb.pairplot(df, hue=target)

X = df[features]
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# StandardScaler needs to be applied to prevent features from having different
# weights during modeling.
ss = StandardScaler()
X_train = pd.DataFrame(ss.fit_transform(X_train), columns=features)
X_test = pd.DataFrame(ss.transform(X_test), columns=features)

# KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto',
# leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None,
# **kwargs)
# n_neighbors=int determines the number of neighbors to take into account.
# weights='distance' multiplies the weight of each neighbor by 1/distance.
# It can also take a function that accepts distance as its argument and returns
# a weight.
# p=int determines the distance metric to use. It is by default 2, or the
# euclidean_distance. p=1 is the manhattan_distance. euclidean_distance is,
# graphically a diagonal straight line between two points. manhattan_distance
# is the sum of the pependicular x and y vectors of a diagonal point.
# euclidean_distance makes more sense in most cases, but manhattan_distance
# may be better for large numbers of features.
# metric=str determines the distance metric to use. By default, p can be set to
# toggle between euclidean_distance and manhattan_distance. However, for a
# string based approach, and more distance metric options, refer to the
# following link:
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', p=2)

print('cross val score:')
print(cross_val_score(knn, X_train, y_train, cv=5))
print()

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(y_pred)
print(y_test)

# confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)
print('confusion matrix:')
print(confusion_matrix(y_test, y_pred))
print()
print('classification report:')
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
