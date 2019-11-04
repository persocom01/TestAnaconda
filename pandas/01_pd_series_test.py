# A Series is a 1d data structure. Basically, a list.
# Demonstrates the various attributes and methods available for use with
# Series.
import pandas as pd

# Also works with a single key dictionary, but the column label causes problems
# with some methods such as s.unique()
data = ['apple', 'banana', 'orange', 'apple', 'pear', 'orange']
s = pd.Series(data)
print(s)

# s.map(self, func_dict_series, na_action=None) returns the series with its
# values maped using the argument passed.
print(s.map({
    'apple': 'pie',
    'banana': 'cake',
    'orange': 'juice',
    'pear': 'candy'
}))
