# A Series is a 1d data structure. Basically, a list.
# Demonstrates the various attributes and methods available for use with
# Series.
import pandas as pd

# Also works with plain lists, but results in no column label.
# Some methods may not work well with column labels, such as s.unique()
data = {'fruits': ('apple', 'banana', 'orange', 'apple', 'pear', 'orange')}
s = pd.Series(data)
print(s)
