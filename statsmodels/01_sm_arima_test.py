import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from datetime import datetime
from datetime import timedelta
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose


import_path = r'.\datasets\electric_production.csv'
# When doing time series modeling, it is common to set the index equal to a
# datetime column.
data = pd.read_csv(import_path, index_col=0)
print(data.head())
print()

# Change to datetime format.
data.index = pd.to_datetime(data.index)
# data.plot()
# plt.show()
# plt.close()

# seasonal_decompose(x, model='additive', filt=None, period=None,
# two_sided=True, extrapolate_trend=0) is the statsmodels library
# implementation of the naive, or classical, decomposition method.
# It breaks down a time series into 4 graphs, observed (or original), trend
# (whether the graph tends to go up or down), seasonal (repeating short term
# cycles), and residual (or noise).
# model can either be 'additive' or 'multiplicative'. Both will give a result,
# so to determine which one to use look at a df.plot() of the observed values.
# If the magnitude of the seasonal variations appear to increase over time,
# it is multiplicative. If they stay the same, it is additive. It is possible
# to transform data into being additive by using sqrt for quadratic trend or
# ln for exponential trend. In practice I fail to see the difference in result.
# Furthermore, more advanced decomposition methods are advised over this one.
# Later versions of statsmodels include STL decomposition under:
# from statsmodels.tsa.seasonal import STL
# There is also the facebook prophet package.
decompose = seasonal_decompose(data, model='multiplicative')
# You may plot the 4 graphs individually by using:
# decompose.observed.plot()
# decompose.trend.plot()
# decompose.seasonal.plot()
# decompose.resid.plot()
decompose.plot()
plt.show()
plt.close()

# We can tell a number of things from the seasonal_decompose plots. A upward
# trend means the data is not stationary, meaning the mean and variance is not
# constant over time. This is needed for ARIMA models having a constant
# expected value unaffected by trend makes it easier to model. We can confirm
# stationarity using the Augmented Dickey-Fuller test.
# adfuller(x, maxlag=None, regression='c', autolag='AIC', store=False,
# regresults=False) tests the null hypothesis that a unit root is present in a
# time series sample. The result of the test is a -ve number. The more -ve,
# the more strongly one can reject the null hypothesis in favor of the
# alternative hypothesis, which is that the sample is stationary.
print('Dickey-Fuller Test:')
# The Dickey-Fuller Test returns a number of things, but we're normally only
# interested in the first two numbers, the test statistic and the p-value.
# If the p-value is < 0.05, we normally accept it as evidence that the time
# series is stationary.
dftest = adfuller(data['energy_production'])
dfoutput = pd.Series(dftest[0:2], index=['Test Statistic', 'p-value'])
print(dfoutput)

data['energy_production_diff1'] = data['energy_production'].diff()
# .diff() produces nan for the first value, which needs to be dropped.
dftest = adfuller(data['energy_production_diff1'].dropna())
dfoutput = pd.Series(dftest[0:2], index=['Test Statistic', 'p-value'])
print(dfoutput)
