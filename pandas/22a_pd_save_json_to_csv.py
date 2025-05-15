# Demonstrates converting json files to csv using pandas
import requests
import pandas as pd

table = 'incident'
file_path = f'./pandas/{table}.csv'
r = requests.get(
    f'https://linkname/{table}',
    auth=('user', 'password'))

if r.status_code == 200:
    data = r.json()
    print(data)
else:
    print(r.json())

normalized_data = pd.json_normalize(data['result'], sep='.')
df = pd.DataFrame(normalized_data)
df.to_csv(file_path, index=False)

# save to tab delimited text instead. Remember to change file_extension to avoid confusion
# df.to_csv(file_path, sep='\t', index=None)