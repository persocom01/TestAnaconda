# The main purpose of grouping is to apply some sort of function to
# rows by their group.
import scipy.stats as stats
import pandas as pd

characters = {
    'name': ['Kazuma', 'Aqua', 'Megumin', 'Darkness', 'Chris', 'Yunyun', 'Wiz', 'Kyouya'],
    'class': ['adventurer', 'priest', 'crusader', 'wizard', 'thief', 'wizard', 'wizard', 'swordmaster'],
    'race': ['human', 'god', 'human', 'human', 'human', 'human', 'lich', 'human'],
    'gender': ['m', 'f', 'f', 'f', 'f', 'f', 'f', 'm'],
    'age': [17, 999, 14, 19, 15, 14, 20, 20],
}
df = pd.DataFrame(characters)
# df.groupby(by, axis=0).
# by accepts arguments used to determine the grouping. It can be a mapping,
# function, label, or list of labels.
grouped = df.groupby(['race', 'gender'])

# Demonstrates iterating though the group.
for group_name, info in grouped:
    print(group_name)
    # Demonstrates choosing a column inside group info.
    print(info['name'])
print()

# Demonstrates selecting a group using the group.get_group() method.
print('get_group:')
print(grouped.get_group(('human', 'f')))
print()

# group.transform(self, func, *args, **kwargs) applies a function to the group.
# It returns a series equal to the size of the group, pretty much allowing you
# to work with the result immediately.
# *args and **kwargs are passed to the function.
print('transform:')
# A number of inbuilt functions exist, that you can pass as strings:
# count / nunique – non-null values / count number of unique values
# min / max – minimum/maximum
# first / last - return first or last value per group
# unique - all unique values from the group
# std – standard deviation
# sum – sum of values
# mean / median / mode – mean/median/mode
# var - unbiased variance
# mad - mean absolute deviation
# skew - unbiased skew
# sem - standard error of the mean
# quantile
print(grouped['age'].transform('mean'))
print()


def get_mode(x):
    output = stats.mode(x)
    return output.count[0]


# group.agg(self, func, *args, **kwargs) allows you to apply multiple functions
# to the group, and even specify different functions for different columns.
# *args and **kwargs don't work if you use multiple functions.
# It returns the same result as appling one of pandas' innate functions
# directly, for example group.mean(), if only one function is applied.
print('agg:')
print(grouped.agg({'gender': get_mode, 'age': 'mean'}))
print()

# group.filter(func, dropna=True) is used to apply a function to a group of
# labels and return any that pass the function criteria.
print('filter:')
print(grouped.filter(lambda x: x['age'].mean() < 21))
print()

# df.pivot_table(self, index, columns, aggfunc, fill_value,
# margins=False, dropna=True, margins_name='All').
# df.pivot_table() is able to groupby() and agg() in a single function.
# Index and column are for the row and column labels, aggfunc is for the
# function to apply, fill_value for empty cells, margins for totals at the
# bottom.
print(df.pivot_table(index='gender', columns=[
      'race'], aggfunc={'age': 'mean'}))
