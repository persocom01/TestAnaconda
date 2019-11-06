# In pandas, NA or nan are considered missing values. NA is a problem during
# loading data from files, because the string 'NA' will be loaded as nan.
import numpy as np
import pandas as pd

import_path = r'.\datasets\null_data.xlsx'

data = pd.read_excel(import_path)
df = pd.DataFrame(data)
print(df)
print(df.isnull().sum())
