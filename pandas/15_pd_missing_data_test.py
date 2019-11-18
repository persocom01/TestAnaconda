# In pandas, NA or nan are considered missing values. NA is a problem during
# loading data from files, because the string 'NA' will be loaded as nan.
import pandas as pd

import_path = r'.\datasets\null_data.xlsx'
# To prevent 'NA' from reading as nan, use na_filter=False.
data = pd.read_excel(import_path)
df = pd.DataFrame(data)
# Note the nan values in the continent column.
print(df)
print()

# Typically .sum() is used as a cell by cell value for isnull is not desirable.
# df.notna() returns the opposite result.
print(df.isnull().sum())
print()

fill_data = {'name': [0, 0, 0], 'continent': ['NA', 0, 0]}
df_fill = pd.DataFrame(fill_data)
# df.fillna(self, value=None, method=None, axis=None, inplace=False,
# limit=None)
# value can be given a dict to target specific columns. If given a series it
# will replace values with values in the series based on shared index.
# A DataFrame will just paste its values over the missing ones if they share
# indices.
# method='ffill', 'bfill' determines how to fill in the gaps. ffill will
# replace a row[1] with a row[0] if row 1 is empty. This will apply even if
# there are multiple empty rows, unless limit=int is used to limit the
# fill effect.
df = df.fillna(df_fill)
print(df)
print()

# df.dropna(self, axis=0, how='any', thresh=None, subset=None, inplace=False)
# Drops rows where nan values are found by default. Set axis=1 for columns.
# how='all' will make it such that the row must be completely empty.
# tresh=int makes it so that row with int number of filled rows will not be
# dropped.
# subset=list_of_labels is a list of columns to check if axis=0 and vice versa.
print(df.dropna(thresh=2))

# df.replace(self, to_replace=None, value=None, inplace=False, limit=None,
# regex=False, method='pad')
# to_replace=str_n_regex_list_dict. If a list is passed, value must also have
# a list of the same length, and a mapping will be performed. When a dictionary
# is passed, the function will map dict keys in the data into the dict values.
# In such a case value should be left empty.
# value=str_n_regex_list_dict. If a dict is passed, the dict keys will be
# treated as DataFrame column names applying different replacement values to
# different columns.
print('replace:')
print(df.replace({'a': 'A', 'b': 'B', 'ab': 'AB'}))
print()
