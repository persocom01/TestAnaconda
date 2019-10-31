import pandas as pd

data = {'fruits': ['apple', 'banana', 'orange', 'apple'],
        'origin': ['usa', 'brazil', 'china', 'china']}
df = pd.DataFrame(data)
print(df)
print(df['fruits'].unique())
print(df['fruits'].str.contains('ora'))
