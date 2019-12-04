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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# For the one_vs_all_roc function.
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from scipy import interp

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


# Demonstrates plotting of multiple ROC curves for a multi categorical target
# in a single plot.


def one_vs_all_roc(y_test, y_pred, average='macro', score_only=False, lw=2, title=None, class_labels=None, **kwargs):
    '''
    A convenience function for plotting Receiver Operating Characteristic (ROC)
    curves or getting the ROC Area Under Curve (AUC) score for multi
    categorical targets.

    class_labels accepts a dictionary of the column values mapped onto class
    names. If the column values are simply integers, it is possible to just
    pass a list.
    '''
    # Gets all unique categories.
    classes = list(set(y_test) | set(y_pred))

    # Converts each multi categorical prediction into a list of 0 and 1 for
    # each category.
    lb_test = label_binarize(y_test, classes=classes)
    lb_pred = label_binarize(y_pred, classes=classes)

    # Returns the mean roc auc score of the multi categorical prediction.
    # The closer it is to 1, the better.
    if score_only:
        return roc_auc_score(lb_test, lb_pred)

    # Compute ROC curve and ROC area for each class.
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i, k in enumerate(classes):
        fpr[k], tpr[k], _ = roc_curve(lb_test[:, i], lb_pred[:, i])
        roc_auc[k] = auc(fpr[k], tpr[k])

    # Initialize graph.
    fig, ax = plt.subplots(**kwargs)

    if average == 'micro' or average == 'both':
        # Compute micro-average ROC curve and ROC area.
        fpr['micro'], tpr['micro'], _ = roc_curve(
            lb_test.ravel(), lb_pred.ravel())
        roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

        ax.plot(fpr['micro'], tpr['micro'], ':r',
                label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})', lw=lw)

    if average == 'macro' or average == 'both':
        # Compute macro-average ROC curve and ROC area.

        # First aggregate all false positive rates.
        all_fpr = np.unique(np.concatenate([fpr[k] for k in classes]))

        # Then interpolate all ROC curves at these points.
        mean_tpr = np.zeros_like(all_fpr)
        for k in classes:
            mean_tpr += interp(all_fpr, fpr[k], tpr[k])

        # Finally average it and compute AUC
        mean_tpr /= len(classes)

        fpr['macro'] = all_fpr
        tpr['macro'] = mean_tpr
        roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

        ax.plot(fpr['macro'], tpr['macro'], ':b',
                label=f'macro-average ROC curve (area = {roc_auc["macro"]:0.2f})', lw=lw)

    # Plot ROC curve for each category.
    colors = cycle(['teal', 'darkorange', 'cornflowerblue'])
    if class_labels is None:
        class_labels = classes
    for k, color in zip(classes, colors):
        ax.plot(fpr[k], tpr[k], color=color,
                label=f'ROC curve of {class_labels[k]} (area = {roc_auc[k]:0.2f})', lw=lw)

    # Plot the curve of the baseline model (mean).
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='best')
    plt.show()
    plt.clf()


one_vs_all_roc(y_test, y_pred, average='both', lw=2, title='species ROC plot',
               class_labels=data.target_names, figsize=(12, 7.5))
