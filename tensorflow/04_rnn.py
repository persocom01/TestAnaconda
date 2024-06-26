# rnn is a neural network designed to process sequential data.
# They are used for purposes such as NLP and Time Series Prediction.
# A basic rnn normally only predicts one step ahead. There are 4 methods in
# which multiple steps ahead can be predicted:
# 1. Multi model strategy.
# Build another model for each step ahead. One for t+1, t+2 and so on.
# 2. Recursive model strategy.
# Use one model to generate t+1, then reuse the output to generate t+2.
# Amplifies errors if this method is used predict far into the future.
# 3. 1+2 hybrid.
# Uses another model for each step, but uses the output for the t+1 model in
# the training data for the t+2 model.
# 4. Multi output strategy.
# Train a single model with multiple outputs. Said to be more complex and
# difficult to train than single output models.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

electricity_prices_file = './datasets/electricity_prices.csv'
gas_prices_file = './datasets/gas_prices.csv'

# load data
electricity_prices = pd.read_csv(electricity_prices_file)
gas_prices = pd.read_csv(gas_prices_file)

# convert date to datetime
electricity_prices['forecast_date'] = pd.to_datetime(electricity_prices['forecast_date'])
electricity_prices['origin_date'] = pd.to_datetime(electricity_prices['origin_date'])
gas_prices['forecast_date'] = pd.to_datetime(gas_prices['forecast_date'])
gas_prices['origin_date'] = pd.to_datetime(gas_prices['origin_date'])

# convert date to day of year
electricity_prices['forecast_dayofyear'] = electricity_prices.apply(lambda x: x['forecast_date'].dayofyear, axis=1)

# merge datetime columns where one is hourly and the other is daily
gas_prices = gas_prices.rename(columns={'origin_date': 'origin_date_gas'})
electricity_price_features = [x for x in electricity_prices.columns if x not in ['forecast_date', 'data_block_id']]
gas_price_features = [x for x in gas_prices.columns if x not in ['forecast_date', 'data_block_id']]
df_merged = pd.merge(electricity_prices[electricity_price_features], gas_prices[gas_price_features], left_on=electricity_prices['origin_date'].dt.date, right_on=gas_prices['origin_date_gas'].dt.date, how='left')
df_merged = df_merged.drop(columns=['key_0', 'origin_date_gas'])
print(df_merged.head())

# setting the target
df_merged['target'] = df_merged['euros_per_mwh'].shift(1)
# remove rows with null values due to the shift
df_merged = df_merged.dropna()
df_merged = df_merged.set_index('origin_date')
features = [col for col in df_merged if not col == 'target']
X = df_merged[features]
y = df_merged[['target']].values

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

# standard scaler
ss = StandardScaler()
# you need to flatten the data to use standard scaler
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.fit_transform(X_test)

n_features = X_train.shape[1]
n_steps = 90

# what TimeseriesGenerator does, is divide the training data into sequences
# comprising the data required to predict the target. For length 60, it means
# 60 previous time steps are taken into account when predicting the target.
# The first sequence will comprise of X[:60], while the first y will be y[60].
# All previous values of y are discarded.
# this function is due to be depreciated. The recommended replacement is to use
# tf.keras.utils.timeseries_dataset_from_array as seen here:
# https://www.tensorflow.org/tutorials/structured_data/time_series
# however, as it is too complicated, a standalone function to produce
# timeseries sequences will be posted below.
train_sequences = tf.keras.preprocessing.sequence.TimeseriesGenerator(X_train_scaled, y_train, length=n_steps, batch_size=64)
test_sequences = tf.keras.preprocessing.sequence.TimeseriesGenerator(X_test_scaled, y_test, length=n_steps, batch_size=64)
print('number of batches:', len(train_sequences))
print()

# klet early stop determine epochs
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='auto')

# define model
# GRU is a type of rnn that attempts to address the vanishing gradient problem,
# thereby being able to capture long term dependencies in sequences better.
# Compared to LSTMs, GRUs are simplier and easier to train.
# GRU has tanh as default activation.
model = tf.keras.models.Sequential([
    tf.keras.layers.GRU(32, input_shape=(n_steps, n_features), return_sequences=True),
    tf.keras.layers.GRU(32),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1)
])

# compile model
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.MeanSquaredError()]
              )

history = model.fit(train_sequences, validation_data=test_sequences, epochs=100)

# model performance
print('model summary')
print(model.summary())


# function for producing timeseries sequences. The result is not exactly the
# same, as it does not return a generator. Instead, the resulting arrays can be
# fed into model.fit as follows:
# history = model.fit(X_train_sequenced, y_train_sequenced, epochs=10, batch_size=32)
# however, validation data cannot be passed.
def split_sequence(sequence, n_steps, y_seq=False):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        x_seq = sequence[i:end_ix]
        X.append(x_seq)
        if y_seq.any():
            y_part = y_seq[end_ix-1]
            y.append(y_part)
    if y_seq.any():
        return np.array(X), np.array(y)
    else:
        return np.array(X)


# evaluate model
X_test_scaled_seq, y_test_seq = split_sequence(X_test_scaled, n_steps, y_test)
loss, mse = model.evaluate(X_test_scaled_seq, y_test_seq, verbose=0)
print('model mse:', mse)

# plots
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Test loss')
plt.legend()
plt.show()
plt.close()

# make predictions
y_pred = model.predict(X_test_scaled_seq)
print(y_pred)
print('prediction:', y_pred[0])
print('actual:', y_test_seq[0])

# plot predictions
plt.plot(y_pred, label='predicted', color='blue')
plt.plot(y_test_seq, label='real', color='orange')
plt.title('electricity price prediction')
plt.xlabel('time steps')
plt.ylabel('electricity price')
plt.legend()
plt.show()
plt.close()


def return_rmse(predicted, test):
    rmse = np.sqrt(mean_squared_error(predicted, test))
    print('the root mean squared error is {:.2f}.'.format(rmse))


return_rmse(y_pred, y_test_seq)
