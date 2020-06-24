# Exponential smoothing is a time series model for data with only a single
# variable. Exponential smoothing exponentially decreases the weight past
# observations based on how long ago they were taken. It comes in 3 flavors:
# simple, double and triple, the difference being whether trend (double) and
# seasonal (triple) variations are also dampened along with the weight of the
# values themselves.
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# prepare data
data = ...
# create class
model = ExponentialSmoothing(data, ...)
# fit model
model_fit = model.fit(...)
# make prediction
yhat = model_fit.predict(...)
