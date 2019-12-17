import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

np.random.seed(1)
df = pd.DataFrame(np.random.randint(0, 10, (30, 2)),
                  index=pd.date_range('1/1/2000', periods=30),
                  columns=['A', 'B'])

# df.rolling(self, window, min_periods=None, center=False, win_type=None,
# on=None, axis=0, closed=None)
# window=int_offset is the number of periods to roll or a time offset, such as
# '3d' for 3 days.
# min_periods=window_size by default. If set lower, you can avoid NaN that
# results when the number of previous data points < window size. min_periods=1
# if a time offset is given for window instead of an int.
# center=True puts the rolled value in the middle of the window instead of the
# end.
# on=col_name of a datetime column if you don't want to roll by DataFrame
# index.
fig, ax = plt.subplots()
ax.plot(df['A'])
ax.plot(df['A'].rolling(window='3d').mean())
# Identical to the above in this case, but the first tow values will be NaN.
ax.plot(df['A'].rolling(window=3).mean())
plt.show()
plt.close()
