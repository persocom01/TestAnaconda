import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from datetime import datetime
from datetime import timedelta
import seaborn as sb
from statsmodels.tsa.statespace.sarimax import SARIMAX
# import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error

import_path = r'.\datasets\stocks_data.csv'
# When doing time series modeling, it is common to set the index equal to a
# datetime column if possible using index_col=0.
data = pd.read_csv(import_path)
# Change to datetime format if necessary.
# data.index = pd.to_datetime(data.index)
data.index = pd.date_range(
    start='1/1/1998', periods=len(data['date']), freq='M')
data.drop(columns='date', inplace=True)
print(data.head())
print()
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
decompose = seasonal_decompose(data['DOW'], model='additive')
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
dftest = adfuller(data['DOW'])
dfoutput = pd.Series(dftest[0:2], index=['Test Statistic', 'p-value'])
# Test Statistic   -2.354857
# p-value           0.154885
print(dfoutput)

data['DOW_d1'] = data['DOW'].diff()
# .diff() produces nan for the first value, which needs to be dropped.
dftest = adfuller(data['DOW_d1'].dropna())
dfoutput = pd.Series(dftest[0:2], index=['Test Statistic', 'p-value'])
# Test Statistic   -1.858603e+01
# p-value           2.075674e-30
print(dfoutput)
print()

# After the time series has been confirmed to be stationary, one can determine
# p and q for the (p, d, q) in ARIMA models using the acf and pacf plots.
# plot_acf(x, ax=None, lags=None, *, alpha=0.05, use_vlines=True,
# unbiased=False, fft=False, missing='none', title='Autocorrelation',
# zero=True, vlines_kwargs=None, **kwargs)
# alpha=0-1 determines the size of the shaded area in the graph. The smaller
# alpha, the larger the area, which represents the confidence interval that the
# null hypothesis that there is no correlation is true.
# acf helps determine q for (p, d, q) of ARIMA.
# As a rule of thumb, q is added when the correlation at lag 1 is -ve, and the
# order of differencing is last bar sticking out of the confidence interval
# closest to the y axis of the acf plot.
# If both acf and pacf show gradual decay, a manual gridsearch ARIMA may be
# needed.
plot_acf(data['DOW_d1'].dropna(), alpha=0.05)
plt.show()
plt.close()
# plot_pacf(x, ax=None, lags=None, alpha=0.05, method='ywunbiased',
# use_vlines=True, title='Partial Autocorrelation', zero=True,
# vlines_kwargs=None, **kwargs)
# method='ywmle' avoids certain errors.
# pacf helps determine p for (p, d, q) of ARIMA.
# As a rule of thumb, p is added when the correlation at lag 1 is +ve, and the
# order of differencing is last bar sticking out of the confidence interval
# closest to the y axis of the pacf plot.
plot_pacf(data['DOW_d1'].dropna(), method='ywmle')
plt.show()
plt.close()

# Train test split by date. It only works because the index has been set to
# datetime format. It works on strings, but with a different result. To ensure
# that the index is already in datetime format, using:
# split_date = pd.datetime(2010, 12, 31)
# will return an error if the DataFrame index is not in datetime format.
# It is possible to split in other ways such as index by taking:
# cutoff = df.shape[0] * 0.8
# train = df.iloc[:cutoff]
# test = df.iloc[cutoff:]
split_date = '2010-01'
train = data['DOW'][:split_date]
test = data['DOW'][split_date:]
# To find the end date.
print(test.tail())
print()

# The correlation at lag 1 is -ve.
order = (0, 1, 2)
# The acf and pacf plots suggest a 16 month seasonality. Since the seasonality
# is +ve, set P=1 and Q=0. Since the seasonal pattern is not stable, set D=0.
seasonal_order = (1, 0, 0, 16)
# SARIMAX(endog, exog=None, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0),
# trend=None, measurement_error=False, time_varying_regression=False,
# mle_regression=True, simple_differencing=False, enforce_stationarity=True,
# enforce_invertibility=True, hamilton_representation=False,
# concentrate_scale=False, trend_offset=1, use_exact_diffuse=False, dates=None,
# freq=None, missing='none', **kwargs)
ar = SARIMAX(train, order=order, seasonal_order=seasonal_order)
model = ar.fit()
# When using AIC as a model selection criterion, the lower the AIC the better.
print(model.summary())

# Also possible to use period numbers. The end in that case would be
# data.shape[0]
forecast = model.predict(start='2010-01', end='2013-08')
# rmse is commonly used to evaluate time series models.
print(forecast)
print('rmse:', np.sqrt(mean_squared_error(test, forecast)))
print()

fig, ax = plt.subplots(figsize=(12, 7.5))
ax.plot(train, label='train')
ax.plot(test, label='test')
ax.plot(forecast, label='forecast')
ax.legend()
plt.show()
plt.close()
