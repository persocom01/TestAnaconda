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


def ordinal_scale(df, mapping=None, start_num=0):
    '''
    A convenience mapping function that accepts a DataFrame and returns it with
    each column defined as keys in the mapping dictionary mapped to its values.
    '''
    if mapping:
        cols = mapping.keys()
        for col in cols:
            df[col] = df[col].map({k: i+start_num for i, k in enumerate(mapping[col])})
            if df[col].isnull().sum() > 0:
                print(f'WARNING: not all values in column "{col}" were mapped.')
    else:
        cols = df.columns
        ord = OrdinalEncoder()
        df[cols] = ord.fit_transform(df[cols])
    return df
