# Demonstrates ways of iterating through pandas DataFrames.
import pandas as pd

characters = {
    'name': ['Kazuma', 'Aqua', 'Megumin', 'Darkness', 'Chris', 'Yunyun', 'Wiz', 'Kyouya'],
    'class': ['adventurer', 'priest', 'crusader', 'wizard', 'thief', 'wizard', 'wizard', 'swordmaster'],
}
df = pd.DataFrame(characters)

# Iteritems
print('iteritems:')
for k, v in df.iteritems():
    print(k)
    print(v)
