# opencv

The xml files for opencv face detectors can be found here: https://github.com/opencv/opencv/tree/master/data

## Installation

## Known issues

1. opencv stores colors in a BGR matrix when an image is loaded. pillow and probably most other applications use RGB order.

2. The coordinates opencv takes while cropping are as follows:

```
im = im[y_top: y_bot, x_left: x_right]
```

By comparison, pillow takes in:

```
im = im.crop(left, top, right, bottom)
```
