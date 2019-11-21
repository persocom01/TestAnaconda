import pandas as pd
import sklearn.model_selection as skms
import sklearn.preprocessing as skpp
import sklearn.compose as skc

features = ['TV', 'radio', 'newspaper']
X = df[features]

poly = skpp.PolynomialFeatures(include_bias=False)
X_poly = poly.fit_transform(X)

poly.get_feature_names(features)

pd.DataFrame(X_poly, columns=poly.get_feature_names(features)).head()
