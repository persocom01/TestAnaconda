# There are three general methods of face detection:
# 1. Haar classifier.
# 2. LBP (Local binary patterns) cascade classifier.
# 3. HOG (Histogram Oriented Gradient features + linear SVM classifier).
# haar classifiers are basically CNNs but with pre-defined convolution filters.
# The main reason to do this is training speed. Haar does works with different
# sized images better than CNN, but performs poorer on slanted images.
# LBP is another method that in its simpliest form, takes a 3x3 grid from the
# image, and condenses the information onto the center pixel as an 8 digit
# binary number representing whether each of the other 8 pixels has a higher or
# lower value than the center pixel. In practice, is is faster than Haar but
# less accurate.
# HOG is similar to LBP in the sense it condenses information into a center
# pixel, but it instead of a binary number, it decomposes the xy surrounding
# pixels into a 2d vector with direction and magnitude. This is repeated for
# every pixel in a cell block (typically 8x8 or 4x4) and the vectors summarized
# as a histogram with 9 bins representing the direction of the vectors. Thus
# a 64 (8x8) pixel grid can be reduced to 9 values. In practice, HOG trains
# much faster than the other two methods, but due to the SVM component,
# comes up with a prediction much slower.
import cv2 as cv
import matplotlib.pyplot as plt
import time

def detect_face(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)

face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
