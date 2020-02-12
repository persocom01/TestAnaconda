# Demonstrates how to load data from files in pandas.
import pandas as pd
import requests
import random
import string
import time

# If you wish to open a file dialog option instead, use:
# filedialog.askopenfilename()
import_path = r'.\datasets\SacramentocrimeJanuary2006.csv'
export_path = r'.\datasets\SacramentocrimeJanuary2006.xlsx'

# pd.read_csv(path, sep=',', header='infer', names=None, index_col=None,
# usecols=None, skiprows=None, skipfooter=0, na_filter=True)
# header=int_list determines the row to use as column label. The first row
# corresponds to header=0.
# names=list allows you to customize the column label. If the number of names
# passed is smaller than the number of columns, the resulting columns will
# contain repeated data columns equivalent to the number of missing names.
# index_col=int_str_list determines the column to use as the row label. Can be
# set to 0 to reuse an index if you did not specify index=False when using
# pandas to save a file.
# usecols=list_callable lets you specify a subset of columns to extract from
# the file instead of extracting the whole thing. Callable is a one argument
# function that will return an int. The order of the list is not taken into
# account.
# skiprows=int_list_callable determines the rows to skip from the top of the
# file, or specific rows if a list is given. Note that it will skip the column
# labels line if you just put skiprows=1.
# skipfooter=int determines the number of rows from the bottom to skip.
# na_filter=True makes the 'NA' string read as nan in the DataFrame. This can
# cause problems, such as when 'NA' actually means North America. To detect
# real nan values while keeping the string NA intact, set:
# keep_default_na=False, na_values=['']
# There are many various other kwargs not discussed here for verbosity.
# pandas accepts other datatypes, including:
# pd.read_excel('xls or xlsx')
# pd.read_table('xml')
# pd.read_json('json')
data = pd.read_csv(import_path, index_col=None, keep_default_na=False, na_values=[''])
df = pd.DataFrame(data)
print(df.head())

# Demonstrates writing the file to excel.
# df.to_excel(self, path, sheet_name='Sheet1')
# Set index=False to avoid the unnamed:0 column that appears if you read the
# file again. Alternatively, set index_col=[0] when reading the file.
# Be warned that excel may truncate long numbers on opening the file. Opening
# the file in python will return the original number as long as you don't save
# over it in excel.
df.to_excel(export_path, index=False)

# Demonstrates how to use requests and pandas together.
url = 'https://swapi.co/api/people/?format=json&search=obi'


def random_word(n):
    return ''.join(random.choice(string.ascii_lowercase) for x in range(n))


# Randomize user agent.
name = f'{random_word(5)} {str(random.randint(0, 10))}.{str(random.randint(0, 10))}'
headers = {'User-agent': name}

r = requests.get(url, params=None, headers=headers)

if r.status_code == 200:
    js = r.json()
else:
    print(r.status_code)

# Demonstrates how to not give yourself away as a bot.
# time.sleep(random.random()*10)

df = pd.DataFrame(js)
print(df)
