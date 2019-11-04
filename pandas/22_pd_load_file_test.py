# Demonstrates how to load data from files in pandas.
import pandas as pd

# If you wish to open a file dialog option instead, use:
# filedialog.askopenfilename()
import_path = r'.\pandas\SacramentocrimeJanuary2006.csv'

# pd.read_csv(path, sep=',', header='infer', names=None, index_col=None,
# usecols=None)
# header=int_or_list determines the row to use as column label. The first row
# corresponds to header=0.
# names=list allows you to customize thethe column label. If the number of
# names passed is smaller than the number of columns, the resulting columns
# will contain repeated data columns equivalent to the number of missing names.
# index_col=int_str_or_list determines the column to use as the row label.
# usecols=list lets you specify a subset of columns to extract from the file
# instead of extracting the whole thing. The order of the list is not taken
# into account.
# There are many various other kwargs not discussed here for verbosity.
# pandas accepts other datatypes, including:
# pd.read_excel('xls or xlsx')
# pd.read_table('xml')
# pd.read_json('json')
data = pd.read_csv(import_path)
print(data.head())
