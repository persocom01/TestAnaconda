# Demonstrates ways of iterating through pandas DataFrames. However, it is
# generally frowned upon to iterate through DataFrames this way due to
# performance issues.
import pandas as pd

characters = {
    'name': ['Kazuma', 'Aqua', 'Megumin', 'Darkness', 'Chris', 'Yunyun', 'Wiz'],
    'class': ['adventurer', 'priest', 'wizard', 'crusader', 'thief', 'wizard', 'wizard'],
}
df = pd.DataFrame(characters)

# df.iteritems is basically the pandas equivalent of python dict.items().
# It returns the column names and a list of values for each column.
print('iteritems:')
for k, v in df.iteritems():
    print(k)
    print(v)
print()

# df.iterrows returns the index and a dictionary of values for each row, the
# dictionary keys being column names. Can generally be replaced with
# df.apply(function, axis=1). The function typically takes as argument a row
# from the DataFrame and returns the row.
print('iterrows:')
for i, v in df.iterrows():
    if i in range(3):
        print(i)
        print(v['name'])
print()

# df.itertuples(self, index=True, name='Pandas') is like iterrows but it
# returns a single named tuple object. Like regular tuples, only integers
# to slice the object.
print('itertuples:')
for t in df.itertuples():
    if t[0] in range(3):
        print(t)
print()
