# Demonstrates the stat functions included in pandas.
import pandas as pd

data = {'col1': [2, 1, 1, 1], 'col2': [1, 3, 2, 4], 'col3': [1, 2, 3, 1]}
df = pd.DataFrame(data)

# Gives a number of dataframe stats, being size, mean, std deviation,
# min, 25th, 50th, 75th percentile and max.
# Not the same as scipy describe().
print(df.describe())
