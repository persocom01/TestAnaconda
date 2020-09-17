# Demonstrates loading an image using opencv.
import glob
from pathlib import Path
import cv2 as cv
import numpy as np
from PIL import Image

infile = './images/innocence.jpg'

im = cv.imread(infile)
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

im = im[int(top): int(bottom), int(left): int(right)]
inp = cv.resize(im, (800, 600))
cv.imshow('image', inp)
cv.waitKey(0)
cv.destroyAllWindows()
