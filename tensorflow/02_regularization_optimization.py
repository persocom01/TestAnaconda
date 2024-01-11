import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# mnist is a dataset consisting of 28x28 grayscale images of digits 0-9.
mnist = tf.keras.datasets.mnist

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print('data shape', X_train.shape)

# standard scaler
ss = StandardScaler()
X_train_flat = X_train.reshape((X_train.shape[0]), -1)
X_test_flat = X_test.reshape((X_test.shape[0]), -1)
X_train_scaled = ss.fit_transform(X_train_flat)
X_test_scaled = ss.fit_transform(X_test_flat)
print('flattened shape:', X_train_scaled.shape)

# define model
model = tf.keras.models.Sequential([
    # as we have already flattened the input
    tf.keras.layers.Dense(128, input_shape=(X_train_scaled.shape[1],), activation='relu'),
    tf.keras.layers.Dense(10)
])
print('model shape:', model.layers[1].output.shape)

# l2 regularization
model2 = tf.keras.models.Sequential([
    # as we have already flattened the input
    tf.keras.layers.Dense(128, input_shape=(X_train_scaled.shape[1],), activation='relu'),
    # there are now 3 types of regularizers in keras, kernel_regularizer,
    # bias_regularizer, and activity_regularizer. In the equation y = mx + c,
    # kernel works on m, bias works on c, and activity works on y. In most
    # cases kernel is sufficient. As a refresher on l1l2 regularization,
    # l1 makes excess features disappear. The list of regularizers can be found
    # here: https://keras.io/api/layers/regularizers/
    tf.keras.layers.Dense(10, kernel_regularizer=tf.keras.regularizers.l2(0.01))
])

# dropout will not be demonstrated as it was done in file 01.

# early stop
# using early stop, you can stop worrying about the optimal number of epochs
# documentation here: https://keras.io/api/callbacks/early_stopping
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, mode='auto')

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
history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=10, batch_size=32)
history2 = model2.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=100, batch_size=32, callbacks=[early_stop])

# model performance
print('model summary')
print(model.summary())
print('model2 summary')
print(model2.summary())

# evaluate model
loss, acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print('model accuracy:', acc)
loss, acc = model2.evaluate(X_test_scaled, y_test, verbose=0)
print('model2 accuracy:', acc)

# losses can be plotted to have a rough idea of how many epochs were needed
plt.plot(history.history['loss'], label='train loss', color='blue')
plt.plot(history.history['val_loss'], label='val loss', color='orange')
plt.title('non regularized')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
plt.close()

plt.plot(history2.history['loss'], label='train loss', color='blue')
plt.plot(history2.history['val_loss'], label='val loss', color='orange')
plt.title('regularized')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
plt.close()

# make predictions
y_pred = model.predict(X_test_scaled[:1])
print(y_pred)
print('prediction:', np.argmax(y_pred))
print('actual:', y_test[:1])
