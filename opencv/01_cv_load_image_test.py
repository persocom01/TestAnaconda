# Demonstrates loading an image using opencv.
import cv2 as cv
import numpy as np
from PIL import Image

# Color image.
# infile = './images/innocence.jpg'
# Grayscale image.
infile = './images/eye.png'

# cv.imread(img) returns a matrix of the image with the color channels in BGR
# order, even if the image is in grayscale. PIL returns the colors of an image
# in RGB order when converted to arrays, and returns an array instead of a
# matrix when used on grayscale images.
im = cv.imread(infile)
pil_img = Image.open(infile)
pil_data = np.asarray(pil_img)
print('opencv BGR: ', im[0][0])
print('pillow RGB: ', pil_data[0][0])

width = im.shape[0]
height = im.shape[1]

m = min(width, height)
m = m*3/4
new_width = m
new_height = m/2

left = (width - new_width)/2
top = (height - new_height)/2
right = (width + new_width)/2
bottom = (height + new_height)/2

# Crop.
im = im[int(top): int(bottom), int(left): int(right)]
inp = cv.resize(im, (800, 600))
# Opencv python stores images in BGR order.
gray = cv.cvtColor(inp, cv.COLOR_BGR2GRAY)
gray = cv.equalizeHist(gray)
# cv.imshow(title, img)
cv.imshow('image', gray)
# Important in when using cv.imshow(title, img) or else the window will close
# immediately.
cv.waitKey(0)
cv.destroyAllWindows()
