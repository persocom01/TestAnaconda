import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# there are generally 5 steps to using tensorflow:
# 1. Define model
# 2. Compile model
# 3. Fit model
# 4. Evaluate model
# 5. Make predictions

# mnist is a dataset consisting of 28x28 grayscale images of digits 0-9.
mnist = tf.keras.datasets.mnist

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print('data shape', X_train.shape)

# standardize data
# there are two ways to standardize data, standard scaler which makes data
# have a mean of 0, and normalization, which makes data range from 0 to 1.
# As a rule of thumb, use normalization for regression problems and standard
# scaler for classification, but normalization can also be used to make the
# modle less sensitive to outliers.

# standard scaler
ss = StandardScaler()
# you need to flatten the data to use standard scaler
X_train_flat = X_train.reshape((X_train.shape[0]), -1)
X_test_flat = X_test.reshape((X_test.shape[0]), -1)
X_train_scaled = ss.fit_transform(X_train_flat)
X_test_scaled = ss.fit_transform(X_test_flat)
print('flattened shape:', X_train_scaled.shape)
# optionally reshape the data back to original.
# X_train_scaled = X_train_scaled.reshape(X_train.shape)
# X_test_scaled = X_test_scaled.reshape(X_test.shape)

# normalize
X_train_normalized, X_test_normalized = X_train / 255.0, X_test / 255.0

# define model
model = tf.keras.models.Sequential([
    # as we have already flattened the input
    tf.keras.layers.Dense(128, input_shape=(X_train_scaled.shape[1],), activation='relu'),
    # dropout prevents overfitting when the sample size is low
    tf.keras.layers.Dropout(0.2)
])
# Check the output shape of the model. layers[index] is unnecessary if you only
# want the final output. The layer index starts from 0.
print('model shape:', model.layers[1].output.shape)
# Add additional layer to model. This is set to 10 here because there are 10
# possible digits in the dataset.
# By default, the output of a dense layer is a logit (multiclass) or
# log odds (binary). This can be changed by a string passed to the activation
# argument. To print out all possible strings, use:
# for k, v in tf.keras.activations.__dict__.items():
#     if not k[0].isupper() and not k[0] == '_':
#         print(k)
# the default activation for dense is linear, Conv2D is relu, and LSTM is
# tanh.
# for the last layer of a neural network, linear activation is suitable for
# regression problems, sigmoid for binary, and softmax for classification.
model.add(tf.keras.layers.Dense(10))

model2 = tf.keras.models.Sequential([
    # What flatten does is reduce an input shape of (28, 28) to
    # (1, 28x28) or (1, 784). We need to do this because subsequent dense
    # layers will work on each row of input individually, so a Dense(128) will
    # have an output of (28, 128) instead of (1, 128).
    tf.keras.layers.Flatten(input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.Dense(128, activation='relu'),
    # dropout prevents overfitting when the sample size is low
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

# compile model
# list of optimizers can be found here: https://keras.io/api/optimizers/
# List of loss functions can be found here: https://keras.io/api/losses/
# List of metrics can be found here: https://keras.io/api/metrics/
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              # SparseCategoricalCrossentropy used here because the result is
              # categorical. If categories are one hot encoded use
              # CategoricalCrossentropy instead.
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              # SparseCategoricalAccuracy is the appropriate metric for
              # SparseCategoricalCrossentropy
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
              )

model2.compile(optimizer=tf.keras.optimizers.Adam(0.001),
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
               )

# fit model
# keras returns an history object, which we can use for plotting
history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=10, batch_size=32)
history2 = model2.fit(X_train_normalized, y_train, validation_data=(X_test_normalized, y_test), epochs=10, batch_size=32)

# evaluate model
loss, acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print('model accuracy:', acc)
loss, acc = model2.evaluate(X_test_normalized, y_test, verbose=0)
print('model2 accuracy:', acc)

# model performance
# the number of parameters is = previous layer x current layer neurons.
print('model summary')
print(model.summary())
print('model2 summary')
print(model2.summary())

# losses can be plotted to have a rough idea of how many epochs were needed
plt.plot(history.history['loss'], label='train loss', color='blue')
plt.plot(history.history['val_loss'], label='val loss', color='orange')
plt.title('scaled')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
plt.close()

plt.plot(history2.history['loss'], label='train loss', color='blue')
plt.plot(history2.history['val_loss'], label='val loss', color='orange')
plt.title('normalized')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
plt.close()

# make predictions
# Also possible to use:
# model(X_test[:1]).numpy()
# Without .numpy() a tensor object will be returned that contains the result,
# shape, and the dtype.
y_pred = model.predict(X_test_scaled[:1])
# in this case, the prediction is returned as a list of logits, which range
# -inf to inf where 0.5 probability corresponds to logit 0.
print(y_pred)
# to get the predicted label, np.argmax is used to get the index of the most
# likely outcome.
print('prediction:', np.argmax(y_pred))
print('actual:', y_test[:1])
