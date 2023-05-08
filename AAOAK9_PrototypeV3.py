# AAOAK9.py
# UC Davis BIM110B 
# Winter 2023
# Group 5: Anisha Kumar, Jasmine Rose Muolic, Blanca Osorio, Marc Ramirez, Emely Rivera 
# Automattic Assessment of Osteoarthritis in Canine Elbows
# Protype V1
import pydicom as dicom
import numpy as np
# import matplotlib as mp
# plt = mp.pyplot
# imread, imshow, show, subplot, title, get_cmap, hist
import math
import skimage
# https://scikit-image.org/docs/stable/api/skimage.filters.html
# use code below to show the image
# plt.imshow(insertImagePixelArrayHere,cmap=plt.cm.gray)
from matplotlib.pyplot import figure, show, imshow, subplot, title, cm, axis
#
def intesityThresh(image,r2,r1=0,setVal = 0): # image = pixels of the image
                                       # m = # of rows
                                       # n = # of columns
                                   # r2 = upper bound threshold
                                   # r1 = lower bound threshold, defaulted to 0 if no input
    thresh = image
    m,n = np.shape(thresh)
    for i in range(0,m):
        for j in range (0,n):
            if image[i][j] >= r1 and image[i][j] <= r2:
                thresh[i][j] = setVal
    return thresh
# Apply a filter N times
def applyNTimes(filt,image,n=1): # filt = filter used
                                 # image = pixels of the image
                                 # n = how many times to apply the filter
    index = 1
    medianImage = filt(image)
    while index < n:
        medianImage = filt(medianImage)
        index += 1
    return medianImage
def findPSNR(array1,array2,maxP = 2**12): # source: https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
    mse = np.mean((array1-array2)**2)
    if mse == 0:
        return 100
    pSNR = 20 * math.log10(maxP/math.sqrt(mse))
    return pSNR
def maxOfArray(array):
    m, n = np.shape(array) # preset dimensions of the radiogrpah when opening them, will help to make code universal for
                                  # different sized radiograph inputs
    tempMax = array[0:m,0]
    indexNum = 0
    for i in array:
        tempMax[indexNum] = max(i)
        indexNum +=1
    compInt = max(tempMax) # intnesity to compoensate for filters, typically scale to values of -1 to 1
    return compInt
def origImageAndApplyFilter(path): # may not need hist eq
    raw = dicom.dcmread(path)
    pixel = raw.pixel_array
    compInt = maxOfArray(pixel) # intnesity to compensate for filters, typically scale to values of -1 to 1
    gaussImage = compInt * skimage.filters.gaussian(pixel,20) # standard deviation of 20
    deblurImage = np.subtract(pixel, gaussImage) # subtract the arrays
    sobelImage = 5 * skimage.filters.sobel(deblurImage)
    sobelResult = np.add(deblurImage,sobelImage)
    maxInt = maxOfArray(sobelResult)
    filtered = intesityThresh(sobelResult,r1 = 2**12-1,r2 = maxInt, setVal = 2**12-1)
    return pixel, filtered
#
path = input('Enter path of the dicom file. Ensure file selected ends in ".dcm": ')
while True: # check if input file is dicom, later will include section outputing the radiograph to check if this
            # is the radiograph the user wants and if their is somethign wrong with the code as well
    if path[-4:] != '.dcm':
        path = input('The file you entered is not a DICOM file. Please re-enter a path leads to a Dicom file.')
    else:
        break
#
orig, filt = origImageAndApplyFilter(path)
figure(figsize=(8,8))
subplot(1,2,1)
imshow(orig, cmap = cm.gray); title('Original')
subplot(1,2,2)
imshow(filt,cmap = 'gray'); title('Filtered') #; axis('off') --> if we want the axis off for later, need to apply for each
show()
#
figure(figsize=(8,8))
imshow(filt,cmap = 'gray'); title('Filtered')
show()