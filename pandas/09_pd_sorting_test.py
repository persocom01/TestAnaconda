# Demonstrates how to sort data in pandas.
import numpy as np
import pandas as pd

unsorted_df = pd.DataFrame(np.random.randn(10, 2), index=[
                           1, 4, 6, 2, 3, 5, 9, 8, 0, 7], columns=['col2', 'col1'])

print(unsorted_df)
