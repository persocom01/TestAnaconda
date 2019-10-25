# Demonstrates how to sort data in pandas.
import numpy as np
import pandas as pd

unsorted_df = pd.DataFrame(np.random.randn(
    5, 3), index=np.random.permutation(5), columns=['col2', 'col1', 'col3'])
print(unsorted_df)

# The first way to sort is by index (1st row or col).
# sort_index(axis=0, ascending=True)
# axis 0 is row, 1 is column.
sorted_df = unsorted_df.sort_index()
print(sorted_df)
print()

data = {'col1': [2, 1, 1, 1], 'col2': [1, 3, 2, 4], 'col3': [1, 2, 3, 1]}
unsorted_df = pd.DataFrame(data)

# The second way is to sort by values.
# sort_values(by, axis=0, ascending=True).
# By takes in the col label if axis=0, row lavel if axis=1.
# You may specify more than one label. In such a case, the first
# label is given priority, then any values in the first col that
# are the same are sorted according to the value of the second label,
# and so on.
sorted_df = unsorted_df.sort_values(by=['col1', 'col2'])
print(sorted_df)
print()

# It does not appear that pandas support sorting by function.
# In such a case, insert a new column containing the function result,
# sort by that column, then remove it afterwards.
