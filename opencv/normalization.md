There are several ways of normalizing an image (in general, a data vector), which are used at convenience for different cases:

Data normalization or data (re-)scaling: the data is projected in to a predefined range (i.e. usually [0, 1] or [-1, 1]). This is useful when you have data from different formats (or datasets) and you want to normalize all of them so you can apply the same algorithms over them. Is usually performed as follows:

Inew = (I - I.min) * (newmax - newmin)/(I.max - I.min)  + newmin
Data standarization is another way of normalizing the data (used a lot in machine learning), where the mean is substracted to the image and dividied by its standard deviation. It is specially useful if you are going to use the image as an input for some machine learning algorithm, as many of them perform better as they assume features to have a gaussian form with mean=0,std=1. It can be performed easyly as:

Inew = (I - I.mean) / I.std
Data stretching or (histogram stretching when you work with images), is refereed as your option 2. Usually the image is clamped to a minimum and maximum values, setting:

Inew = I
Inew[I < a] = a
Inew[I > b] = b
Here, image values that are lower than a are set to a, and the same happens inversely with b. Usually, values of a and b are calculated as percentage thresholds. a= the threshold that separates bottom 1% of the data and b=the thredhold that separates top 1% of the data. By doing this, you are removing outliers (noise) from the image. This is similar (simpler) to histogram equalization, which is another used preprocessing step.

Data normalization, can also be refereed to a normalization of a vector respect to a norm (l1 norm or l2/euclidean norm). This, in practice, is translated as to:

Inew = I / ||I||
where ||I|| refeers to a norm of I.

If the norm is choosen to be the l1 norm, the image will be divided by the sum of its absolute values, making the sum of the whole image be equal to 1. If the norm is choosen to be l2 (or euclidean), then image is divided by the sum of the square values of I, making the sum of square values of I be equal to 1.

The first 3 are widely used with images (not the 3 of them, as scaling and standarization are incompatible, but 1 of them or scaling + streching or standarization + stretching), the last one is not that useful. It is usually applied as a preprocess for some statistical tools, but not if you plan to work with a single image.
