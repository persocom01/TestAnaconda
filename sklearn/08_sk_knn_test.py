# Demonstrates the k nearest neighbors algorithm which is mainly used for
# categorical target predictive problems. It can be used on a continuous target
# as well, but there are probably better models for that. It works well on
# datasets with fewer features. It is unknown if it works on multiple targets.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pleiades as ple

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
# At this point a parameter search using GridSearchCV is normally used to
# determine the best hyperparameters to use, although it is not done here.
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', p=2)

# Plots error rate vs k value in a graph. You can just avoid this step when
# using GridSearchCV, but can be good for visualization purposes to know the
# rough range of k to use.


def knn_k_error_plot(X_train, y_train, y_test, k=10, **kwargs):
    error_rate = []
    for i in range(1, k):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        error_rate.append(np.mean(y_pred != y_test))

    fig, ax = plt.subplots(**kwargs)
    ax.plot(range(1, k), error_rate, color='blue', linestyle='dashed',
            marker='o', markerfacecolor='red', markersize=10)
    ax.set_xlabel('K')
    ax.set_ylabel('Error Rate')
    ax.set_title('Error Rate vs. K Value')
    plt.show()
    plt.close()


knn_k_error_plot(X_train, y_train, y_test)

print('cross val score:')
print(cross_val_score(knn, X_train, y_train, cv=5))
print()

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
y_prob = knn.predict_proba(X_test)
print(y_pred)
print(y_test)

yuri = ple.Yuri()

# classification_report(y_true, y_pred, labels=None, target_names=None,
# sample_weight=None, digits=2, output_dict=False)
# Precision is the percentage of positive predictions that were correct.
# Recall is the percentage of positive outcomes that were correctly predicted.
# f1-score is a combination of both following the formula:
# 2*(Recall * Precision) / (Recall + Precision)
# Support is the total number of true outcomes of each class.
# Macro average is just the mean score.
# Weighted average is the mean sore with each category weighted by support.
# target_names=list_dict accepts a list of target names or a dictionary in the
# form {newlabel: oldlabel}
# digits=int determines the decimal places.
# output_dict=True returns the output as a dict, where individual values can be
# referenced via dict[str_label_name][metric_name]
print('classification report:')
print(classification_report(y_test, y_pred, output_dict=False))
print()
# confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)
# By default, the y_pred are the columns of the matrix, y_true are the rows,
# arranged from smallest to largest from left to right. If strings are used,
# they will be in alphabetical order. You may determine the order by passing a
# list to the labels argument.
# sample_weight=list must be the same size as y_pred. It determines the value
# of each prediction relative to the others as it is shown in the matrix.
# By default, all predictions are worth 1.
# The number of true predictions will be the sum of the diagonal.
print('confusion matrix:')
print(confusion_matrix(y_test, y_pred))
print()

# Due to the unrealistically high accuracy of the model, th ROC for the model
# is 1. This is not a bug.
yuri.plot_roc(y_test, y_prob, figsize=(12.5, 7.5))
