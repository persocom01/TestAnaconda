# Demonstrates string methods in pandas.
import pandas as pd
import re

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

# Applies python split(seperator, max) to all rows. It turns all strings into
# lists of strings even if no split was actually performed.
print('split:')
print(s.str.split('|')[0][0])
print()

# s.str.cat(self, others, sep=dividor, na_rep=none, join='none').
# Joins the strings and optionally other series or strings into a single one.
# na_rep determines what nan will be replaced with.
# join determines the join style if there is another series, and in future will
# be made left by default.
print('cat:')
print(s.str.cat(sep='_'))
print()

# s.str.get_dummies(self, sep='|', drop_first=False) turns a series of strings
# into a DataFrame containing one hot vectors. The reason to do this appears to
# be that some operations cannot be performed on strings, making it easier to
# process them if they are turned into numbers first.
# When dealing with one hot vector, you generally drop one as one of them has
# to be made a point of reference. For example, if gender can only be male or
# female, it makes no sense to have one hot vectors for both, since a 0 in male
# will always be a 1 in female, and vice, versa.
# pd.get_dummies(data, prefix=None, prefix_sep='_', dummy_na=False,
# columns=None, sparse=False, drop_first=False, dtype=None) is a more in depth
#  function.
print('one hot vectors:')
df = s.str.get_dummies(sep='|')
print(df)
print()

# s.str.contains(self, pattern, case=True, flags=0, na=nan, regex=True)
# Used to check a series of strings to see if any contains a certain pattern.
# flags are regex flags, na is what to replace nan with.
print('contains:')
print(s.str.contains(r'APPLE', flags=re.I))
print()

# s.str.replace(self, pattern, replacement, n=-1, case=None, flags=0,
# regex=True)
# n is the number of replacements made. -1 = all.
print('replace:')
print(s.str.replace(r'APPLE', 'watermelon', flags=re.I))
print()

# s.str.repeat(n) repeats each string n times.
print('repeat:')
print(s.str.repeat(2))
print()

# s.str.count(pattern, flags=0)
# Counts the number of occurances of pattern in each string.
print('count:')
print(s.str.count('A', flags=re.I))
print()

# s.str.startswith(self, pat, na=nan)
# Returns a boolean for each string.
# s.str.endswith(self, pat, na=nan) does the same thing but for the end.
print('starts with:')
print(s.str.startswith('apple'))
print()

# s.str.find(self, sub, start=0, end=None)
# Returns 0 on success and -1 on failure.
print('find:')
print(s.str.find('apple'))
print()

# s.str.findall(self, pattern, flags=0)
# Returns a list containing all strings that match the pattern for all strings
# in the series.
print('findall:')
print(s.str.findall(r'A\w+e', flags=re.I))
print()

# s.str.swapcase(self)
print('swapcase:')
print(s.str.swapcase())
print()

# s.str.islower(self)
# Checks if all characters in a string are lowercase and returns a boolean.
# Does this for all strings in the series.
# s.str.isupper(self) does the same but uppercase.
# s.str.istitle(self) does the same but titlecase.
print('islower:')
print(s.str.islower())
print()

# s.str.isnumeric(self)
# Checks if all characters in a string are numeric, meaning they can be
# converted to int or float and returns a boolean.
# Ironically, returns nan if it checks a number instead of a string.
# Other checks include:
# s.str.isalpha(self)
# s.str.isalnum(self) for alphanumeric. It doesn't appear to accept anything
# except letters and numbers.
print('isnumeric:')
print(s.str.isnumeric())
print()
