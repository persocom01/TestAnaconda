import pandas as pd

data = {
    'col1': [2, 1, 1, 1],
    'col2': [1, 3, 2, 4],
    'col3': [1, 2, 3, 1]
}
df = pd.DataFrame(data)
print(df.pct_change())
