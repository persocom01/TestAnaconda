import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as skds
import sklearn.feature_selection as skfs
import sklearn.model_selection as skms
import sklearn.linear_model as sklm
import sklearn.metrics as skm

# Use this command if using Jupyter notebook to plot graphs inline.
# %matplotlib inline

data = skds.load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
print(df.head())
print()

features = data.feature_names
X = df[features]
y = data.target

X_train, X_test, y_train, y_test = skms.train_test_split(X, y, random_state=1, stratify=data.target)

np.set_printoptions(precision=2)
skb = skfs.SelectKBest(score_func=skfs.chi2, k=4)
skb.fit(X_train, y_train)
print(skb.scores_)
