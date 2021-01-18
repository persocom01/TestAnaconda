# Tensorflow

A tensorflow testing playground based on the tutorials found here:
1. https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/
2. https://www.tensorflow.org/tutorials/quickstart/beginner

## Installation

tensorflow comes with Anaconda. To install it separately, enter into command line:

```
pip install tensorflow
```

Once installed, one can confirm the version of tensorflow with the following python script:

```
import tensorflow as tf
print(tf.__version__)
```

It is important to differentiate tensorflow 1.x.x from tensorflow 2.x.x due to the many changes in code that are the result of integration with keras.

## Usage

Using a model in tensorflow.keras can be divided into 5 steps:
1. Define model
2. Compile model
3. Fit model
4. Evaluate model
5. Make predictions

### Define model

```
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])
```

Choice of [layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers).

The choice of [activation function](https://www.tensorflow.org/api_docs/python/tf/keras/activations) is as follows:
1. sigmoid (sigmoid)
0-1 output most commonly used for binary classification problems.
2. relu (ReLU or rectified linear unit)
0-inf output that has been the standard for intermediate layers.
3. tf.keras.layers.LeakyReLU(alpha=0.3) (leaky ReLU)
Same as ReLU but prevents the "dying ReLu" problem by making -ve weights possible. However, "dying ReLU" is not always undesirable, as it can speed up model optimization by reducing complexity. In tensorflow, it has only been recently possible to implement this as an activation function. Implemented as a layer otherwise.
4. linear
Output for linear regression problems.
5. tanh
-1-1 output, which is basically a sigmoid that can go -ve. Was the standard activation function for intermediate layers before relu, but can still be used so long as the input is normalized between -1-1. Due to the fact the mean output of tanh is around 0, it trains faster than sigmoid.
6. softmax
0-1 outputs which sum to 1 for multiclass problems.
7. swish
A relu-like activation function, whose main difference lies in that small -ve outputs are not treated as zero, but large -ve outputs are. Said to perform better than relu.
8. gelu (GELU or Gaussian Error Linear Unit)
gelu looks similar to swish. It seems to have gained a lot of popularity from 2019-2021 onwards and is used in a number of state of the art (SOTA) NLP models.

### Compile model

```
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
```

Optimizers can be found here: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers

The choice of [loss function](https://www.tensorflow.org/api_docs/python/tf/keras/losses) is as follows:
1. keras.losses.binary_crossentropy (binary cross entropy)
For binary classification problems.
2. keras.losses.categorical_crossentropy (categorical cross entropy)
For multiclass problems.
3. keras.losses.sparse_categorical_crossentropy (sparse categorical cross entropy)
A more memory efficient version of categorical cross entropy for instances when all classes are mutually exclusive, or when the number of classes is very large. The main difference is it produces the predicted class and probability instead of all possibly classes and their probabilities.
4. keras.losses.poisson (poisson loss)
For problems where the dataset has a poisson distribution. It maximizes the likelyhood that the target comes from a poisson distribution.
5. keras.losses.mean_squared_error (mean squared error)
For regression problems. Causes large errors to be more heavily penalized compared to other measures of loss.
6. keras.losses.mean_absolute_percentage_error (mean absolute percentage error)
For regression problems. An intuitive measure of error in cases when understandability is more important.
7. keras.losses.huber_loss (huber loss, also known as smooth mean absolute error)
For regression problems. Huber loss can be thought of as a modified version of mean absolute error. Huber loss smooths out the loss curve with a much gentler slope about the origin, instead of a constant gradient, making it easier to find the loss minima. Huber loss can be twerked using the delta value, which is normally made between 10 to 0.1. Lower values penalize large errors less.
8. tf.keras.losses.log_cosh (logarithm of the hyperbolic cosine)
For regression problems. Log-cosh is similar to huber loss but has the added advantage of being twice differentiable everywhere. This makes it better for certain ML frameworks like XGBoost.
9. keras.losses.mean_squared_logarithmic_error (mean squared logarithmic error)
For regression problems. Used when the target range is large, thus penalizing large errors is undesirable. Has the special property of penalizing underestimates of the target more than overestimates, which can be desirable in quoting prices.
10. keras.losses.cosine_similarity (cosine similarity)
Returns a value between -1 and 1, where 1 indicates dissimilarity and -1 similarity. 0 is the perpendicular.

tensorflow metrics functions can be found here: https://www.tensorflow.org/api_docs/python/tf/keras/metrics

### Fit model

```
model.fit(X, y, epochs=100, batch_size=32, verbose=2)
```

### Evaluate model

```
loss = model.evaluate(X, y, verbose=0)
```

### Make predictions

```
y_pred = model.predict(X)
```
