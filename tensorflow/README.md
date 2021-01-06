# Tensorflow

A tensorflow testing playground based on the tutorial found here: https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/

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
model =
```

Activation functions can be found here: https://www.tensorflow.org/api_docs/python/tf/keras/activations

The choice of activation function is as follows:
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
6. softmax - 0-1 outputs which sum to 1 for multiclass problems.
7. swish
A modified relu, where the main difference lies in that small -ve outputs are not treated as zero, but large -ve outputs are. Said to possibly perform better than relu.
8. gelu (GELU or Gaussian Error Linear Unit)
A version of ReLU that looks like swish, but is more curvy about the origin. It gained a lot of popularity from 2019-2021 onwards and is used in a number of state of the art (SOTA) models.

### Compile model

```
model.complie(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
```

Optimizers can be found here: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers

tensorflow loss functions can be found here: https://www.tensorflow.org/api_docs/python/tf/keras/losses

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
