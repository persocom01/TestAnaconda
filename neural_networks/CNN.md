CNNs are a type of NN that are often specifically used in image recognition. It can also be used in time series and NLP.

The reason being the size of images such as 1024 x 1024 results in a huge number of features. CNN solves this by detecting patterns in pixels close to each other instead of taking the image as a whole, and once these patterns are detected, the number of features is further compressed through pooling.

During the convolution (filtering) stage, the convolution matrix is equal to (n-f+1)x(n-f+1), where the input matrix is (n, n) and filter matrix is (f, f). The filter matrix is normally 3x3 or 5x5.

We may also use padding, or 0s around the image, of size (f-1)/2 in order to preserve edge information of the image.

stride is the size of the step the filter takes to make the result matrix. A large step results in a smaller result. We still need an overlap during convolution.

During pooling, we compress the convolution matrix without overlap with a 2x2 pooling matrix, which halves the dimensions. For this reason we should try and have a convolution matrix with even dimensions. How we compress it can be adjusted, but we can choose things like max value in the pooling matrix. Average also exists but is uncommonly used.
