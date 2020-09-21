import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import os
import time

save_X = './TFJS face detection/data/X.npy'
save_y = './TFJS face detection/data/y.npy'
X = np.load(save_X)
y = np.load(save_y)
X = X[:5000]
y = y[:5000]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

X_train = tf.expand_dims(X_train, 3)
X_test = tf.expand_dims(X_test, 3)

X_val = X_train[:500]
y_val = y_train[:500]

lenet_5_model = keras.models.Sequential([
    keras.layers.Conv2D(6, kernel_size=5, strides=1,  activation='tanh', input_shape=X_train[0].shape, padding='same'), #C1
    keras.layers.AveragePooling2D(), #S2
    keras.layers.Conv2D(16, kernel_size=5, strides=1, activation='tanh', padding='valid'), #C3
    keras.layers.AveragePooling2D(), #S4
    keras.layers.Flatten(), #Flatten
    keras.layers.Dense(120, activation='tanh'), #C5
    keras.layers.Dense(84, activation='tanh'), #F6
    keras.layers.Dense(1, activation='sigmoid') #Output layer
])

lenet_5_model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

root_logdir = os.path.join(os.curdir, 'logs\\fit\\')


def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

lenet_5_model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val), callbacks=[tensorboard_cb])
lenet_5_model.evaluate(X_test, y_test)
