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

# NLTK.
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer

# Feature engineering.
from sklearn.preprocessing import PolynomialFeatures

# Model selection.
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

# Linear models.
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LogisticRegression

# KNN.
from sklearn.neighbors import KNeighborsClassifier

# Naive bayes.
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

# Decision trees.
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBClassifier

# SVM
from sklearn.svm import SVC
from sklearn.svm import SVR

# Metrics.
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# For the cramers_corr function
import scipy.stats as stats
from itertools import combinations


def vif_feature_select(df, max_score=5.0, inplace=False, drop_list=False, _drops=None):
    '''
    Takes a DataFrame and returns it after recursively eliminating columns
    with the highest VIF scores until the remainder have a VIF scores of less
    than max_score.

    drop_list=True gets a list of features that would be dropped instead.
    '''
    # Avoids overwriting the original DataFrame by default.
    if inplace is False:
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

    np.fill_diagonal(corr_matrix, 1.0)
    df_corr_matrix = pd.DataFrame(corr_matrix, index=cols, columns=cols)

    return df_corr_matrix
