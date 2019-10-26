# Demonstrates merging DataFrames in pandas.
# Merging is different from concat because the df are not just
# added to each other, but instead, are united into one based on
# a shared key.
import pandas as pd

data = {
    'id': [1, 2, 3, 4, 5],
    'name': ['apple', 'banana', 'grape', 'lemon', 'papaya']
}
data2 = {
    'id': [1, 2, 3, 4, 5],
    'taste': ['sweet', 'sweet', 'sweet', 'sour', 'sweet']
}
left_df = pd.DataFrame(data)
print(left_df)
