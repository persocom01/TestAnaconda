# A Series is a 1d data structure. Basically, a list.
# Demonstrates the various attributes and methods available for use with
# Series.
import pandas as pd

# Also works with a single key dictionary, but the column label causes problems
# with some methods such as s.unique()
data = ['apple', 'banana', 'orange', 'apple', 'orange']
s = pd.Series(data)
print(s)
print()

# s.value_counts(self, normalize=False, sort=True, ascending=False, bins=None,
# dropna=True) counts the number of times each unique value occurs in the
# series. Useful for counting categorical variables.
print('value counts:')
vc = s.value_counts()
print(vc)
# Sometimes it is helpful to save the output to a file.
vc.reset_index().to_csv('./pandas/value_counts.csv', index=False)
print()

# s.duplicated(keep='first')
# Used to find duplicated rows.
# keep=first/last/False determines which duplicate is considered false so it
# can be kept. False causes all duplicates to return True.
print('duplicated:')
print(s.duplicated())
print()
