# Demonstrates how to load data from files in pandas.
import pandas as pd

# If you wish to open a file dialog option instead, use:
# filedialog.askopenfilename()
import_path = r'.\pandas\SacramentocrimeJanuary2006.csv'
export_path = r'.\pandas\SacramentocrimeJanuary2006.xlsx'

# pd.read_csv(path, sep=',', header='infer', names=None, index_col=None,
# usecols=None, skiprows=None, skipfooter=0)
# header=int_list determines the row to use as column label. The first row
# corresponds to header=0.
# names=list allows you to customize the column label. If the number of names
# passed is smaller than the number of columns, the resulting columns will
# contain repeated data columns equivalent to the number of missing names.
# index_col=int_str_list determines the column to use as the row label.
# usecols=list_callable lets you specify a subset of columns to extract from
# the file instead of extracting the whole thing. Callable is a one argument
# function that will return an int. The order of the list is not taken into
# account.
# skiprows=int_list_callable determines the rows to skip from the top of the
# file, or specific rows if a list is given. Note that it will skip the column
# labels line if you just put skiprows=1.
# skipfotter=int determines the number of rows from the bottom to skip.
# There are many various other kwargs not discussed here for verbosity.
# pandas accepts other datatypes, including:
# pd.read_excel('xls or xlsx')
# pd.read_table('xml')
# pd.read_json('json')
data = pd.read_csv(import_path)
df = pd.DataFrame(data)
print(df.head())

# df.to_excel(self, path, sheet_name='Sheet1')
df.to_excel(export_path)
