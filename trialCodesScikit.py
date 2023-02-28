# trial codes, getting familiar with sckit image
import cv2
import pydicom as dicom
import numpy as np
import matplotlib as mp
#
from skimage import data, io, filters
camera = data.camera()
print(camera)
print('Dimensions =',camera.shape)
print('Size =',camera.size)
#
print('Min and Max',camera.min(), camera.max())
print('Pixel Mean intesnity =',camera.mean())
#
print('Pixel 10th row 20th column =',camera[10, 20])
## NumPY indexing
#
# Slicing
camera[:10] = 0 # set first 10 lines in 0, black
# Masking
mask = camera < 87
camera[mask] = 255 # where mask is true, set the value to 255
# intensity thresholding
#
#
print('Length Camera =',len(camera)) # len = 512
inds_r = np.arange(len(camera)) # creates an array 1 x input, so [1 x 512]
inds_c = 4 * inds_r % len(camera)
print(inds_r) # [0,1,2,3,...,511]
print(inds_c) # [0,4,8,12,...,508]
# will create a stripe pattern like the spiraling on a candy cane
#
camera[inds_r, inds_c] = 0 # goes array index by index for each and assigns the
                           # value with 0
print('Pixel(1,4)',camera[1,4]) # ie. this is 0
#
# Disk Region of interest
nrows, ncols = camera.shape
row, col = np.ogrid[:nrows, :ncols]
cnt_row, cnt_col = nrows / 2, ncols / 2
outer_disk_mask = ((row - cnt_row)**2 + (col - cnt_col)**2 >(nrows / 2)**2)
camera[outer_disk_mask] = 0 # creates the circle outline the with diameter the
                            # size of orignial image
#
# these 2 lines are to show the image w/ skimage
io.imshow(camera)
io.show()
