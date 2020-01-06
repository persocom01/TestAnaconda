# Demonstrates how to use autoarima to find the optimal parameters for a
# SARIMAX model. autoarima can then be used to make the predictions by itself,
# or one can use the parameters to initialize a statsmodels SARIMAX model.
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

import_path = r'.\datasets\stocks_data.csv'
# When doing time series modeling, it is common to set the index equal to a
# datetime column if possible using index_col=0.
data = pd.read_csv(import_path)
data.index = pd.date_range(
    start='1998-01-01', periods=len(data['date']), freq='M')
data.drop(columns='date', inplace=True)
print(data.head())
print()

data['INTC_l1'] = data['INTC'].shift(1)

split_date = pd.datetime(2010, 1, 1)
train = data[:split_date]
test = data[split_date:]

# auto_arima(y, exogenous=None, start_p=2, d=None, start_q=2, max_p=5, max_d=2,
# max_q=5, start_P=1, D=None, start_Q=1, max_P=2, max_D=1, max_Q=2,
# max_order=5, m=1, seasonal=True, stationary=False,
# information_criterion='aic', alpha=0.05, test='kpss', seasonal_test='ocsb',
# stepwise=True, n_jobs=1, start_params=None, trend=None, method='lbfgs',
# maxiter=50, offset_test_args=None, seasonal_test_args=None,
# suppress_warnings=False, error_action='warn', trace=False, random=False,
# random_state=None, n_fits=10, return_valid_fits=False, out_of_sample_size=0,
# scoring='mse', scoring_args=None, with_intercept=True, sarimax_kwargs=None,
# **fit_args)
# seasonal=True fits a SARIMA.
# trace=True will print status of fits.
# n_jobs=-1 uses all cores in the case stepwise=False, but that is not
# recommended.
offset = 1
aar = auto_arima(train['DOW'].iloc[offset:], exogenous=train[['INTC_l1']].iloc[offset:], start_p=1, d=1, start_q=1, max_p=3,
                 max_d=1, max_q=3, start_P=0, D=None, start_Q=0, m=16, seasonal=True, error_action='ignore', suppress_warnings=True, trace=True)
model = aar.fit(train['DOW'])
print(model.summary())

# Unlike SARIMAX, you don't need to provide the exogenous variable to make a
# prediction.
pred = model.predict(n_periods=data.shape[0]-train.shape[0])
pred = pd.DataFrame(pred, index=test.index, columns=['prediction'])

fig, ax = plt.subplots(figsize=(12, 7.5))
ax.plot(train['DOW'], label='train')
ax.plot(test['DOW'], label='test')
ax.plot(pred, label='pred')
ax.legend()
plt.show()
plt.close()
