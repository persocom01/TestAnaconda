import pandas as pd
import datetime as dt

# demonstrates the merging of a daily an an hourly column
df_hourly = pd.DataFrame({
    'date_hourly': pd.date_range(start='2023-01-01', end='2023-01-10', freq='H'),
    'values1': range(217)
    })
df_daily = pd.DataFrame({
    'date_daily': pd.date_range(start='2023-01-01', end='2023-01-10', freq='D'),
    'values2': range(10)
    })

df_merged = pd.merge(df_hourly, df_daily, left_on=df_hourly['date_hourly'].dt.date, right_on=df_daily['date_daily'].dt.date, how='left')
df_merged = df_merged.drop(columns=['key_0', 'date_daily'])
print('merging daily and hourly')
print(df_merged.head())
print()

print('current time:', dt.datetime.now())
# Timestamp is pandas' replacement for python's datetime object.
# You may give it a datetime like string in YY-MM-DD HH:MM:SS.NS or give it
# or a number + unit, unit being 'D' for days, 'h' hours and so on.
print('timestamp:', pd.Timestamp('2017-03-01'))
print()

# pd.date_range(start=None, end=None, periods=None, freq=None, tz=None,
# normalize=False, name=None)
# Returns a list of dates between start and end. Number of dates depends on the
# period or freq arguments.
# frequency accepts various strings as arguments, the list can be found here:
# https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
# end includes the end. periods is the size of the list returned.
# Of the first 4 arguments, only 3 need to be given, for example if start, end,
# and periods are given, the resulting frequency will be evently spaced.
# tz is the timezone, for example 'Asia/Hong_Kong'.
# normalize sets the start/end dates to midnight.
# name sets the name attribute of the resulting object.
dates = pd.date_range(pd.Timestamp('2019-11-1'), pd.Timestamp('2019-11-5'), 5)
print('date range:')
print(dates)
print()

# pd.bdate_range(start=None, end=None, periods=None, freq='B', tz=None,
# normalize=True, name=None, holidays=None) is pd.date_range() which skips
# sat and sun by default (freq='B').
# holidays=list_of_dates can be passed to skip even more days.
dates = pd.bdate_range(pd.Timestamp('2019-11-1'), periods=5)
print('bdate range:')
print(dates)
print()

# pd.to_datetime(arg, errors='raise', dayfirst=False, yearfirst=False,
# format=None, exact=True, unit=None, origin='unix', cache=True)
# errors can also be set to 'coerce', which sets them to NaT or 'ignore', which
# will return the input.
# dayfirst and yearfirst can be set to account for ambiguous date formats.
# By default, the function assumes MM-DD-YY.
# format allows you to set how the datetime is read from the input. Details:
# https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
# exact=False allows format to match anywhere in the string instead of exactly.
print('to datetime:')
dates_std = pd.to_datetime(dates, errors='ignore', format='MM-DD-YY')
s_dates = pd.Series(dates_std)
print(s_dates)
print()

# Once in datetime format, various attributes of datetime can be accessed.
print('year:')
print(s_dates.dt.year[0])
print('quarter of the year:')
print(s_dates.dt.quarter[0])
print('month:')
print(s_dates.dt.month[0])
print('day:')
print(s_dates.dt.day[0])
print('hour:')
print(s_dates.dt.hour[0])
print('minute:')
print(s_dates.dt.minute[0])
print('second:')
print(s_dates.dt.second[0])
