import pandas as pd
import sklearn.datasets as skds
import sklearn.preprocessing as skpp

import_path = r'.\datasets\drinks.csv'

data = pd.read_csv(import_path)
df = pd.DataFrame(data)
print(df.head())

# skpp.MinMaxScaler(feature_range=(0, 1), copy=True)
