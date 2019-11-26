import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

features = ['TV', 'radio', 'newspaper']
X = df[features]

poly = PolynomialFeatures(include_bias=False)
X_poly = poly.fit_transform(X)

poly.get_feature_names(features)

pd.DataFrame(X_poly, columns=poly.get_feature_names(features)).head()
