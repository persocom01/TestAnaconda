# Demonstrates more in depth use of regex in pandas.
import pandas as pd
import re

# Generally, pandas text methods work with a series or other form of 1d array.
# It can be a natural series or just a df[col_name].
data = {
    'name': ['Kazuma', 'Aqua', 'Megumin', 'Darkness'],
    'age': [17, None, 14, 19],
    'sex': ['M', 'F', 'F', 'F'],
    'class': ['adventurer', 'arch priest', 'arch wizard', 'crusader']
    }
df = pd.DataFrame(data)
print(df)
print()

# Extract first 5 characters of every name:
# extract() requires regex to be put in groups, even if only 1 groups is used.
# This is because additional groups will be returned as additional columns.
df['first_five_Letter'] = df['name'].str.extract(r'(^\w{5})')
print('extracting the first 5 characters of every name:')
print(df)
df.pop('first_five_Letter')
print()

# Return series satisfying condition:
print('characters with more than one a in their names:')
# str.lower() used to convert to lowercase first.
print(df[df['name'].str.lower().str.count(r'a') > 1])
print('count:', df['sex'].str.count(r'^F.*').sum())
print()

# match() returns a boolean, so you don't need to set a condition.
print('only female characters:')
print(df[df['sex'].str.match(r'F')])
print()

# Demonstrates regex replace with groups.
print('remove arch from class names:')
df['class'] = df['class'].replace(r'arch (.*)', r'\1', regex=True)
print(df)
print()
