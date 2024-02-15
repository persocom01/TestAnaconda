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

stock_prices_file = './datasets/stocks_data.csv'

# load data
stock_prices = pd.read_csv(stock_prices_file)
print(stock_prices.head())


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


n_steps = 60
features = 1
# split into samples
X_train, y_train = split_sequence(training_set_scaled, n_steps)
