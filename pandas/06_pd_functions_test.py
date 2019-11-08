# Demonstrates how to apply functions to pandas DataFrames.
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
# args and kwargs are passed to the function.
print(df[['name', 'origin']].pipe(suffix_adder, '_suff'))
print()

# s.map(self, func_dict_series, na_action=None) returns the series with its
# values maped using the argument passed. Unlike apply or pipe, it does apply
# the function to each cell individually.
print('map:')
print(df['name'].map({
    'apple': 'pie',
    'banana': 'cake',
    'orange': 'juice',
}))
print()


def shortfall(df):
    return df['supply'] - df['demand']


# df.apply(self, func, axis=0, raw=False, result_type=None, args=(), **kwds)
# applies a function to a series, or in the case of a DataFrame, divides the
# DataFrame into series based on the axis argument and applies the function
# to it. It does not apply the function to each cell individually, and as such,
# the function must be workable on the series object and not individual values.
# For simpler operations like this one, it is simplier use:
# df['shortfall'] = df['supply'] - df['demand']
print('apply:')
df['shortfall'] = df.apply(shortfall, axis=1)
print(df)
print()


def lower_case(s, case='lower'):
    return str(s).lower()


# df.applymap(self, func) actually applies a function to all cells in the
# DataFrame. Strangely, it can't pass any arguments to the function.
print('applymap:')
print(df.applymap(lower_case))
print()
