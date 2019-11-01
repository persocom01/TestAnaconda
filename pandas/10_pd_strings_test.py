# Demonstrates string methods in pandas.
import pandas as pd

# Generally, pandas text methods work with a series or other form of 1d array.
# It can be a natural series or just a df[col_name].
data = ['apple', 'banana', '3.apple|orange']
s = pd.Series(data)

print('lower/upper/title:')
print(s.str.lower())
print(s.str.upper())
print(s.str.title())
print()

print('capitalize/swapcase:')
# 'Sentence' casing.
print(s.str.capitalize())
print(s.str.swapcase())
print()

# Returns the length of each string in the series, and not string size, unlike
# python len().
print('len:')
print(s.str.len())
print()

# s.str.strip(self, to_strip=none) strips \n \t and whitespaces from the start
# and end of strings by default. An argument can be given to get it to strip
# other characters instead.
# To strip only from the start, use s.str.lstrip(self, to_strip=none).
# To strip only from the end, use s.str.rstrip(self, to_strip=none).
print('strip:')
s = s.str.strip('3.')
print(s)
print()

print('split:')
print(s.str.split('|'))
print()

# s.str.cat(self, others, sep=dividor, na_rep=none, join='none').
# Joins the strings and optionally other series or strings into a single one.
# na_rep determines what nan will be replaced with.
# join determines the join style if there is another series, and in future will
# be made left by default.
print('cat:')
print(s.str.cat(sep='_'))
print()

# s.str.get_dummies(self, sep='|') turns a series of strings into a categories
# containing one hot vectors. The reason to do this appears to be that some
# operations cannot be performed on strings, making it easier to process them
# if they are turned into numbers first.
print('one hot vectors:')
print(s.str.get_dummies())
print()

# print(df['fruits'].str.contains('ora'))
