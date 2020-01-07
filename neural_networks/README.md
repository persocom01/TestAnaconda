3 main frameworks:

1. Keras - wrapper on Tensorflow and other NN.
2. Tensorflow - by Google.
3. PyTorch - by FaceBook, good at NLP.

NN layers generally have 2**n nodes.

They also need to be tuned:
1. Number of layers - relationships become more complex.
2. Number of nodes per layer - more relationships.
3. Choice of activation functions.
4. Loss function.
5. Regularization.

The choice of activation functions, including:
1. sigmoid - 0-1 output for output in binary classification problems.
2. ReLU (rectified linear unit) - 0-inf the most common function for layers.
3. leaky ReLU - same as ReLU but prevents dead nodes by making -ve weights possible.
4. identity/linear - no change for output in regression problems.
5. tanh - -1-1, same as Sigmoid by can go -ve. Tends to perform worse than Relu for layers and same as Sigmoid for output.
6. softmax - 0-1 outputs which sum to 1 for multiclass problems.

NN are fit using gradient descent using your choice of loss functions:
1. binary_crossentropy - binary classification problems.
2. categorical_crossentropy - multiclass problems.
3. mse - regression problems.
4. etc

Gradient descent uses certain algorithms to optimize convergence. We normally use Adam. RMSProp is also common. Others can be found here:
https://ruder.io/optimizing-gradient-descent/index.html#gradientdescentoptimizationalgorithms

The algorithms use 3 general model training methods:
1. stochastic (random) - takes 1 row and updates weights n row times.
2. batch - takes n rows and updates weights 1 time.
3. minibatch - takes n/x rows and updates weights x times. Minibatches normally have size the power of 2.

We also specify the number of times it runs through the data attempting to find convergence, called epochs.

Regularizations:
1. Penalty parameters in the layers.
2. Dropout.
3. Early stopping.
