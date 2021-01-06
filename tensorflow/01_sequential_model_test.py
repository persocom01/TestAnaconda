import tensorflow as tf

# mnist = tf.keras.datasets.mnist
#
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train, X_test = X_train / 255.0, X_test / 255.0

# define the model
model = tf.keras.models.Sequential([
    # What flatten does is reduce an input shape of (28, 28) to
    # (1, 28x28) or (1, 784). We need to do this because subsequent dense
    # layers will work on each row of input individually, so a Dense(128) will
    # have an output of (28, 128) instead of (1, 128).
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2)
])
# Check the output shape of the model. layers[index] is unnecessary if you only
# want the final output. The layer index starts from 0.
print(model.layers[2].output.shape)
# Add additional layer to model.
# By default, the output of a dense layer is a logit (multiclass) or
# log odds (binary). To change this an appropriate activation
model.add(tf.keras.layers.Dense(10))
