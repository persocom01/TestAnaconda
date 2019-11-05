import pandas as pd

print('current time:', pd.datetime.now())
# Timestamp is pandas' replacement for python's datetime object.
# You may give it a datetime like string in YY-MM-DD HH:MM:SS.NS or give it
# or a number + unit, unit being 'D' for days, 'h' hours and so on.
print('timestamp:', pd.Timestamp('2017-03-01'))
# pd.date_range(start=None, end=None, periods=None, freq=None, tz=None, normalize=False, name=None, closed=None, **kwargs)
# end includes the end. periods is the size of the list returned.
# Of the first 4 arguments, only 3 need to be given, for example if start, end,
# and periods are given, the resulting frequency will be evently spaced.
# tz is the timezone, for example 'Asia/Hong_Kong'.
# normalize sets the start/end dates to midnight.
# name sets the name attribute of the resulting object.
#
dates = pd.date_range(pd.Timestamp('2019-11-1'), pd.Timestamp('2019-11-5'), 5, closed='right')
print(dates)
# print(pd.date_range("11:00", "13:30", freq="H").time)
print()

dates = ['05.12.08 15:30:45', 'Thu 5 dec 2019 1530 45s']

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
dates_std = pd.to_datetime(dates)
s_dates = pd.Series(dates_std)
print(s_dates)
print()

# Once in datetime format, various attributes of datetime can be accessed.
print('year:')
print(s_dates.dt.year)
print('quarter of the year:')
print(s_dates.dt.quarter)
print('month:')
print(s_dates.dt.month)
print('day:')
print(s_dates.dt.day)
print('hour:')
print(s_dates.dt.hour)
print('minute:')
print(s_dates.dt.minute)
print('second:')
print(s_dates.dt.second)

# df[col].dt.month == 1


# print(pd.Timestamp(1587687255, unit='s'))

#
