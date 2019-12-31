import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import seaborn as sb
import matplotlib.pyplot as plt

register_matplotlib_converters()

np.random.seed(1)
df = pd.DataFrame(np.random.randint(0, 10, (30, 2)),
                  index=pd.date_range('1/1/2000', periods=30),
                  columns=['A', 'B'])

# df.rolling(self, window, min_periods=None, center=False, win_type=None,
# on=None, axis=0, closed=None) takes the stat function on top of it, applies
# it to window number of rows, and returns the result to the last row in the
# window. It is used to calculate stuff like moving averages.
# window=int_offset is the number of periods to roll or a time offset, such as
# '3d' for 3 days.
# min_periods=window_size by default. If set lower, you can avoid NaN that
# results when the number of previous data points < window size. min_periods=1
# if a time offset is given for window instead of an int.
# center=True puts the rolled value in the middle of the window instead of the
# end.
# on=datetime_col_name if you don't want to roll by DataFrame index.
fig, ax = plt.subplots()
ax.plot(df['A'])
ax.plot(df['A'].rolling(window='3d').mean(), label='rolling mean 3d')
# Identical to the above in this case, but the first two values will be NaN.
ax.plot(df['A'].rolling(window=3).mean(), label='rolling mean 3')
ax.legend()
plt.show()
plt.close()

# df.expanding(self, min_periods=1, center=False, axis=0) takes the stat
# function on top of it, applies it to min periods number of rows, and returns
# the result to the last row. Subsequent rows add themselves to the end of this
# calculation without taking away the beginning. The result is a cumsum for
# sum(), but functions like mean() get very gradual.
fig, ax = plt.subplots()
ax.plot(df['A'])
ax.plot(df['A'].expanding(min_periods=3).sum(), label='expanding sum')
ax.plot(df['A'].expanding(min_periods=3).mean(), label='expanding mean')
ax.legend()
plt.show()
plt.close()

# df.ewm(self, com=None, span=None, halflife=None, alpha=None, min_periods=0,
# adjust=True, ignore_na=False, axis=0) calculates the exponential moving
# average. The parameters determine how important the most recent data point
# is. Either com, span or halflife must be specified.
print(df['A'][:5])
print(df['A'].ewm(com=0.5, min_periods=3).mean()[:5])
fig, ax = plt.subplots()
ax.plot(df['A'])
ax.plot(df['A'].rolling(window=3).mean(), label='rolling mean 3')
ax.plot(df['A'].ewm(com=0.5, min_periods=3).mean(),
        label='exponential moving mean 3')
ax.legend()
plt.show()
plt.close()

# df.diff(self, periods=1, axis=0)
# diff =  current - previous(with periods number of lags)
# period=int sets the number of periods to lag. Setting it higher doesn't take
# the sum of more previous periods, it simply takes a period from further back.
# Used in time series ARIMA analysis to find d in (p, d, q). A period more than
# 1 can be used to get a seasonal difference, such as 12 for annual seasonality
# in monthly periods.
fig, ax = plt.subplots()
ax.plot(df['A'])
ax.plot(df['A'].diff(2), label='diff 2')
ax.legend()
plt.show()
plt.close()
