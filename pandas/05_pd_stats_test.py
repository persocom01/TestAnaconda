# Demonstrates the stat functions included in pandas.
import numpy as np
import pandas as pd

# np.NaN is used to represent an empty cell.
# nan is treated as non existant and not 0.
data = {'col1': [-2, 1, 1, 1], 'col2': [1, 3, np.NaN, 4], 'col3': [1, 2, 3, 1]}
df = pd.DataFrame(data)

# Returns number of non nan cells per column.
print('count:')
print(df.count())
print()
print('sum:')
print(df.sum())
print()
print('mean:')
print(df.mean())
print()
print('median:')
print(df.median())
print()
# Will return multiple nodes if all of them occur the same number of times.
print('mode:')
print(df.mode())
print()
print('std deviation:')
print(df.std())
print()
print('min:')
print(df.min())
print()
print('max:')
print(df.max())
print()
print('abs:')
print(df.abs())
print()
print('product:')
print(df.prod())
print()
print('cumulative sum:')
print(df.cumsum())
print()
print('cumulative product:')
print(df.cumprod())
print()

# Gives a number of DataFrame stats, being size, mean, std deviation, min,
# 25th, 50th, 75th percentile and max.
# If the DataFrame comprises only strings, count, unique and mode
# (top and freq) will be given instead.
# Not the same as scipy describe().
print('describe:')
print(df.describe())
print()
