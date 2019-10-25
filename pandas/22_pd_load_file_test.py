# Demonstrates how to load data from files in pandas.
import pandas as pd

# If you wish to open a file dialog option instead, use:
# filedialog.askopenfilename()
import_path = r'.\pandas\SacramentocrimeJanuary2006.csv'
# pandas accepts other datatypes, including:
# pd.read_excel('xls or xlsx')
# pd.read_table('xml')
# pd.read_json('json')
data = pd.read_csv(import_path)
print(data)
