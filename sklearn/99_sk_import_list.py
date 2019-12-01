import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Datasets.
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_digits
from sklearn.datasets import load_linnerud
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer

# Preprocessing.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OrdinalEncoder

# Feature engineering.
from sklearn.preprocessing import PolynomialFeatures

# Model selection.
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import RandomizedSearchCV

# Linear models.
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LogisticRegression

# Metrics.
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# KNN.
from sklearn.neighbors import KNeighborsClassifier

# For the one_vs_all_roc function.
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from scipy import interp

# For the cramers_corr function
import scipy.stats as stats
from itertools import combinations


def ordinal_scale(df, mapping=None, start_num=0):
    '''
    A convenience mapping function that accepts a DataFrame and returns it with
    each column defined as keys in the mapping dictionary mapped to its values.
    '''
    if mapping:
        cols = mapping.keys()
        for col in cols:
            df[col] = df[col].map(
                {k: i+start_num for i, k in enumerate(mapping[col])})
            if df[col].isnull().sum() > 0:
                print(
                    f'WARNING: not all values in column "{col}" were mapped.')
    else:
        cols = df.columns
        ord = OrdinalEncoder()
        df[cols] = ord.fit_transform(df[cols])
    return df


def vif_feature_select(df, max_score=5.0, inplace=False, drop_list=False, _drops=None):
    '''
    Takes a DataFrame and returns it after recursively eliminating columns
    with the highest VIF scores until the remainder have a VIF scores of less
    than max_score.

    drop_list=True gets a list of features that would be dropped instead.
    '''
    # Avoids overwriting the original DataFrame by default.
    if not inplace:
        df = df.copy()
    # Creates an empty list for the first iteration.
    if _drops is None:
        _drops = []
    features = df.columns
    # VIF is the diagonal of the correlation matrix.
    vifs = np.linalg.inv(df.corr().values).diagonal()
    max_vif_index = np.argmax(vifs)
    # By default, the function only takes into account the VIF score when
    # eliminating features.
    if vifs[max_vif_index] >= max_score:
        _drops.append(features[max_vif_index])
        del df[features[max_vif_index]]
        return vif_feature_select(df, max_score, inplace, drop_list, _drops)
    else:
        # Returns a list of features that would be dropped instead of a
        # DataFrame
        if drop_list:
            return _drops
        else:
            return df


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

def cramers_corr(df):
    '''
    Takes a DataFrame of categorical variables and returns a DataFrame of the
    correlation matrix based on the Cramers V statistic. Uses correction from
    Bergsma and Wicher, Journal of the Korean Statistical Society 42 (2013):
    323-328

    Does not require variables to be label encoded before use.
    '''

    def cramers_v(x, y):
        con_matrix = pd.crosstab(x, y)
        chi2 = stats.chi2_contingency(con_matrix)[0]
        n = con_matrix.sum().sum()
        phi2 = chi2/n
        r, k = con_matrix.shape
        phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
        rcorr = r-((r-1)**2)/(n-1)
        kcorr = k-((k-1)**2)/(n-1)
        return np.sqrt(phi2corr/min((kcorr-1), (rcorr-1)))

    cols = df.columns
    n_cols = len(cols)
    corr_matrix = np.zeros((n_cols, n_cols))
    for col1, col2 in combinations(cols, 2):
        i1, i2 = cols.get_loc(col1), cols.get_loc(col2)
        corr_matrix[i1, i2] = cramers_v(df[col1], df[col2])
        corr_matrix[i2, i1] = corr_matrix[i1, i2]

    df_corr_matrix = pd.DataFrame(corr_matrix, index=cols, columns=cols)

    return df_corr_matrix
