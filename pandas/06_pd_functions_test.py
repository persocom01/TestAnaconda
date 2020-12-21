# Demonstrates how to apply functions to pandas DataFrames.
import numpy as np
import pandas as pd

data = {
    'name': ['apple', 'banana', 'orange', 'apple', 'orange'],
    'origin': ['USA', 'brazil', 'china', 'china', 'australia'],
    'supply': [361, 851, 418, 439, 279],
    'demand': [264, 833, 516, 371, 194]
}
df = pd.DataFrame(data)


def suffix_adder(df, suffix):
    return df + str(suffix)


# df.pipe(self, func, *args, **kwargs) applies a function to the whole
# DataFrame. It does not apply the function to the cells individually. However,
# certain operations like df ** 2 or df + str can be performed as if they were
# applied to every cell in the DataFrame.
# Because the returned object is a dataframe, pipe can be chained to apply
# multiple functions to the same dataframe.
# args and kwargs are passed to the function.
print(df[['name', 'origin']].pipe(suffix_adder, '_suff').pipe(suffix_adder, '_suff2'))
print()

# s.map(self, func_dict_series, na_action=None) returns the series with its
# values maped using the argument passed. Unlike apply or pipe, it applies
# the function to each cell individually.
print('map:')
print(df['name'].map({
    'apple': 'pie',
    'banana': 'cake',
    'orange': 'juice',
}))
# Demonstrates adding a feature to the DataFrame using map.
asia = ['china', 'japan', 'korea']
df['from_asia'] = df['origin'].map(lambda x: 1 if x in asia else 0)
print(df)
print()

# df.apply(self, func, axis=0, raw=False, result_type=None, args=(), **kwds)
# applies a function to a series, or in the case of a DataFrame, divides the
# DataFrame into series based on the axis argument and applies the function
# to it. The default axis treats each column in the dataframe as a series.
# It does not apply the function to each cell individually, and as such, the
# function must be workable on the series object and not individual values.
# For simpler operations, it is easier to use:
# df['result'] = df['supply'] - df['demand']
# df[col_name].apply(pd.Series) can be used to unpack nested dictionaries.
print('apply:')
# log transforms are often used to make exponential graphs linear.
df['log transform'] = df['supply'].apply(lambda x: np.log(x))
print(df[['supply', 'demand']].apply(lambda x: x.max() - x.min()))
print()


def lower_case(s, case='lower'):
    return str(s).lower()


# df.applymap(self, func) is the DataFrame version of map, and applies a
# function to all cells in the DataFrame.
print('applymap:')
print(df.applymap(lower_case))
print()
