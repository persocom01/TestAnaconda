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
# you need to flatten the data to use standard scaler
X_train_flat = X_train.reshape((X_train.shape[0]), -1)
X_test_flat = X_test.reshape((X_test.shape[0]), -1)
X_train_scaled = ss.fit_transform(X_train_flat)
X_test_scaled = ss.fit_transform(X_test_flat)

# reshape the data to 28 x 28 x 1 because it is a 28 x 28 picture that is
# monochrome. If we expect RGB colors use 3 instead of 1.
X_train_scaled = X_train_scaled.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test_scaled = X_test_scaled.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
print('data reshaped:', X_train_scaled.shape)

# demonstrates converting y into one hot encoded categories.
# There does not appear to be a reason to do this when we can use
# SparseCategoricalCrossentropy instead of CategoricalCrossentropy to handle
# one hot encoding internally during the model compilation stage, but this is
# demonstrated as it follows the GA tutorial.
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
print('to category:', y_train[:10])

# define model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]), activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# cnn
model2 = tf.keras.models.Sequential([
    # in a cnn the first number is the number of filters. The more complex the
    # data, the more filters may be needed.
    # kernel_size = int_or_tuple determines the size of the smallest part of
    # an image the model will attempt to capture patterns of. A small
    # kernel_size (~3) will capture finer details such as edges while a large
    # kernel_size (5-7) will detect larger patterns. However, larger
    # kernel_size costs more computation power.
    tf.keras.layers.Conv2D(8, kernel_size=3, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2], X_train_scaled.shape[3]), activation='relu'),
    # pool_size reduces the feature map by the pool size factor. In the
    # case of 2, a 64x64 feature map is reduced to 32x32. Larger pool sizes
    # (3-4) may be more helpful in detecting harger patterns, just like larger
    # kernel_size.
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # deeper in a cnn it is common to have more filters to capture abstract
    # and complex features.
    tf.keras.layers.Conv2D(16, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# compile model
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              # as we one hot encoded the data beforehand, we will use
              # CategoricalCrossentropy() here instead of
              # SparseCategoricalCrossentropy()
              loss=tf.keras.losses.CategoricalCrossentropy(),
              # you may use either Accuracy() or CategoricalAccuracy() as
              # metric. The difference being that CategoricalAccuracy()
              # calculates accuracy separately for each class before averaging
              # them, which can be useful if the problem has imbalanced
              # classes. It also gives more 'sensible' accuracy values.
              metrics=[tf.keras.metrics.Accuracy()]
              )

model2.compile(optimizer=tf.keras.optimizers.Adam(0.001),
               loss=tf.keras.losses.CategoricalCrossentropy(),
               metrics=[tf.keras.metrics.Accuracy()]
               )

# fit model
# keras returns an history object, which we can use for plotting
history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=10, batch_size=32)
history2 = model2.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=10, batch_size=32)

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
plt.title('FFNN')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
plt.close()

plt.plot(history2.history['loss'], label='train loss', color='blue')
plt.plot(history2.history['val_loss'], label='val loss', color='orange')
plt.title('CNN')
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
