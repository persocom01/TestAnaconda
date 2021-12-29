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
print('extracting the first 5 characters of every name:')
df['first_five_Letter'] = df['name'].str.extract(r'(^\w{5})')
print(df)
df.pop('first_five_Letter')
print()

# Return series satisfying condition:
print('characters with more than one a in their names:')
# str.lower() used to convert to lowercase first.
print(df[df['name'].str.lower().str.count(r'a') > 1])
print('count:', df['sex'].str.count(r'^F.*').sum())
print()

# .match() returns a boolean, so you don't need to set a condition. However, it
# only matches the beginning of the string. To match anywhere in a string, use
# .contains() instead.
print('only female characters:')
print(df[df['sex'].str.match(r'F')])
print()

# .match() returns a boolean, so you don't need to set a condition. However, it
# only matches the beginning of the string. To match anywhere in a string, use
# .contains() instead.
print('only priests:')
print(df[df['class'].str.contains(r'priest', flags=re.IGNORECASE, regex=True, na=False)])
print()

# Demonstrates pandas inbuilt regex .replace() with groups.
print('remove arch from class names:')
df['class'] = df['class'].replace(r'arch (.*)', r'\1', regex=True)
print(df)
print()

# Use .finall() when you do not want the original series returned, but a new
# list of only the elements found. Probably not suitable if you still want to
# return a datqaframe. For instance, if we want it to do the same thing
# .match() did, it looks as follows:
print('only female characters, but with findall:')
print(df[[True if len(e) > 0 else False for e in df['sex'].str.findall(r'F')]])
print()

# .split() changes strings into lists of strings according to where they were
# split. It splits by whitespace by default.
# n=int limits the number of times split is performed.
# expand=False can be set to true if you want to split the column into new
# columns such as follows:
print('split names into at most 3 parts before and after the letter a:')
col_names = ['name_part1', 'name_part2', 'name_part3']
n = len(col_names)
col_names_dict = {x: col_names[x] for x in range(n)}
df = pd.concat([df, df['name'].str.split(r'[aA]', n=n-1, expand=True).rename(columns=col_names_dict)], axis=1)
print(df)
print()
