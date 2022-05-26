# Demonstrates prasing and reformating dates from DataFrames.
# https://www.w3schools.com/python/gloss_python_date_format_codes.asp
import pandas as pd
import datetime as dt


# Demonstrates a custom date parser. While pandas is capable of parsing dates
# without it, this can lead to problems, especially when month and day are
# confused. The default date format in pandas is YYYY-MM-DD (%Y-%m-%d).
def date_parser(date):
    try:
        # print(str(date))
        # date = datetime.datetime.strptime(str(date), '%m/%d/%Y %H:%M:%S')
        date = dt.datetime.strptime(str(date), '%Y-%m-%d')
        return date
    except Exception:
        return pd.NaT


data = {
    'date': ['12/05/2010 00:00:00', '07/16/2010 00:00:00', None, '27/11/2010 00:00:00'],
    'sheep': [1, 3, 2, 4]
    }
date_col = ['date']
df = pd.DataFrame(data, parse_dates=date_col, date_parser=date_parser)


print(df[date_col[0]])
df[date_col[0]] = df[date_col[0]].dt.strftime('%Y-%m-%d')

print(df)
