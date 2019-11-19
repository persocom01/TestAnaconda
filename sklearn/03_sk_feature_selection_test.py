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
