# In pandas, NA or nan are considered missing values. NA is a problem during
# loading data from files.
import numpy as np
import pandas as pd

data = {'col1': [2, 1, 1, np.nan], 'col2': [1, 3, 2, 4]}
df = pd.DataFrame(data)
print(df)
print(df.isnull())
