# Tests for the pillow image object.
import os
from PIL import Image

infile = r'.\images\Reddit.jpg'
# Image.open(fp, mode='r')
im = Image.open(infile)
# Image.show(title=None, command=None)
# im.show()

# img.format returns None if image is not read from a file.
print('extension:', im.format)
print('width, height:', im.size)
# Returns 1 for bilevel images and L for greyscale. Bilevel are images with
# only 2 colors, normally represented as black and white. They are represented
# by a single bit, 0 or 1, so no color in between exists.
print('monochrome or RGB:', im.mode)

# Demonstrates converting files to another extension.
f, e = os.path.splitext(infile)
outfile = f + ".png"
print(outfile)
if infile != outfile:
    try:
        with Image.open(infile) as im:
            # save saves the file in a format according to its extension.
            # The list of recognized formats can be found here:
            # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html
            # If you wish to save a file in a format without using the default
            # extension, the format argument must be used.
            im.save(outfile, format='jpeg')
    except OSError:
        print('cannot convert', infile)

# When cropping, the region is defined by a rectangle defined by a pair of xy
# coordinates (left_x, left_y, right_x, right_y), where the upper left of the
# image is considered to have coordinates (0, 0).
width, height = im.size
min = min(im.size)
new_width = min
new_height = min

# Center image.
left = (width - new_width)/2
top = (height - new_height)/2
right = (width + new_width)/2
bottom = (height + new_height)/2

# Image.crop(box=None)
center_box = (left, top, right, bottom)
region = im.crop(center_box)
# region.show()

# Image.paste(im, box=None, mask=None) pastes an image onto another.
# box can be given as (left_x, left_y) or (left_x, left_y, right_x, right_y).
# If all 4 coordinates are given, the size of paste_image must match the size
# of the box it is pasted into.
# mask is an optional parameter between 255 and 0. It controls the transparency
# of the pasted image, with 255 being opaque.
region = region.transpose(Image.ROTATE_180)
# Unlike .crop, .paste does not accept floats in the argument for box. To get
# the sizes to match exactly, use crop_box  = [int(round(x)) for x in crop_box]
center_box = [int(round(x)) for x in center_box]
im.paste(region, center_box)
# im.show()

# Image.split() converts rgb images into 3 greyscale images representing the
# intensity of their colors, with white being the most intense.
r, g, b = im.split()
# g.show()
# Demonstrates flipping the r and b color bands.
im = Image.merge("RGB", (b, g, r))
# im.show()

# Image.rotate(angle, resample=0, expand=0, center=None, translate=None,
# fillcolor=None) The difference between transpose and rotate is first that
# rotate doesn't rotate the dimensions of the image itself. This means a 50x100
# image will still be 50x100 after rotation, with the blank spaces filled with
# black. Secondly, rotate doesn't only accept 90 degree intervals.
# expand=1 makes the rotate effect similar to transpose.
# center=(x, y) defines the locus of rotation. (0, 0) being the top left. Set
# to the center of the image by default.
# fillcolor=color determines the color to fill the blank spaces that may result
# after rotation. It appears to accept strings with common colors like 'red'
# 'green' 'blue', but more importantly accepts a (red, green, blue) tuple of
# integers between 0 and 255 for color customization.
im = im.rotate(45, fillcolor=(0, 100, 0))
# im.show()
# img.resize will cause an image to be stretched or squashed depending on its
# original dimensions.
im = im.resize((min, min))
# im.show()

# Image.transpose(method) is used for 90 degree image rotations as well as
# flipping the vertical or horizonal axes of the image.
im = region.transpose(Image.ROTATE_90)
# im.show()
im = im.transpose(Image.FLIP_LEFT_RIGHT)
# im.show()
im = im.transpose(Image.FLIP_TOP_BOTTOM)
# im.show()

# Image.convert(mode=None, matrix=None, dither=None, palette=0, colors=256)
# is used to convert images from RGB to greyscale and stuff like that.
# mode=mode is the most important argument. Use L for greyscale and RGB for
# color. The list of modes can be found here:
# https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
# dither is FLOYDSTEINBERG by default. The effect of which can described as
# making color gradients appear less choppy.
im = im.convert('L')
im.show()
