# Demonstrates applying aggregations to the DataFrame, which
import numpy as np
import pandas as pd

data = {
    'col1': [2, 1, 1, 1],
    'col2': [1, 3, 2, 4],
    'col3': [1, 2, 3, 1]
}
df = pd.DataFrame(data)

r = df.rolling(window=3, min_periods=1)
print(r.aggregate(np.sum))
