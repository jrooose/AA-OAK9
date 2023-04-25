# AAOAK9.py
# UC Davis BIM110B 
# Winter 2023
# Group 5: Anisha Kumar, Jasmine Rose Muolic, Blanca Osorio, Marc Ramirez, Emely Rivera 
# Automattic Assessment of Osteoarthritis in Canine Elbows
# Protype V1
import cv2
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
from matplotlib.pyplot import imread, imshow, show, subplot, subplots, title, hist, cm, gray
#
#
def intesityThresh(image,m,n,r2,r1=0): # image = pixels of the image
                                       # m = # of rows
                                       # n = # of columns
                                   # r2 = upper bound threshold
                                   # r1 = lower bound threshold, defaulted to 0 if no input
    thresh = image
    for i in range(0,m):
        for j in range (0,n):
            if image[i][j] >= r1 and image[i][j] <= r2:
                thresh[i][j] = 0
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
#
# R1
path = '/Users/marcramirez/Desktop/AAOAK9/Anonymous Rad Pt 2/R1/Elbow_Lf_Rf - 145421/Right_Fore_CR_CD_Extremity_10/IM-0003-0001.dcm'
# This path is based off of my computer so just change it to be the path of R1/Elbow_Lf_Rf - 145421/Right_Fore_CR_CD_Extremity_10 
# Will prompt user to input their file name later on
pathTest = '/Users/marcramirez/Desktop/AAOAK9/Anonymous Rad Pt 2/R2/Thoracic_Limb - 59409/L_AP_Elbow_6/IM-0003-0001.dcm'
# same view but left foot
#
while True: # check if input file is dicom, later will include section outputing the radiograph to check if this
            # is the radiograph the user wants and if their is somethign wrong with the code as well
    if path[-4:] != '.dcm':
        path = input('The file you entered is not a DICOM file. Please re-enter a path leads to a Dicom file.')
    else:
        break
#
rawImage = dicom.dcmread(path)
pixelsRaw = rawImage.pixel_array # 12-bit
imshow(pixelsRaw,cmap=cm.gray); title('Orignial Joint')
show()
#
testRaw = dicom.dcmread(pathTest)
pixelTest = testRaw.pixel_array # 12-bit
imshow(pixelsRaw,cmap=cm.gray); title('Orignial Joint')
show()
#
onlyJoint = pixelsRaw[400:1200,300:900] # segmentation by hand
compInt = maxOfArray(onlyJoint) # intnesity to compensate for filters, typically scale to values of -1 to 1
compIntTest = maxOfArray(pixelTest)
print('MaxComp=',compInt)
print('MaxCompTest',compIntTest)
imshow(onlyJoint, cmap = cm.gray); title('Manually Segmented Joint')
show()
#
# Unsharp Masking (Deblurring)
gaussImage = compInt * skimage.filters.gaussian(onlyJoint,20) # standard deviation of 20
deblurImage = np.subtract(onlyJoint, gaussImage) # subtract the arrays
#imshow(onlyJoint,cmap=cm.gray); title('Orignial')
#show()
#imshow(gaussImage,cmap=cm.gray); title('Gaussian(sigma = 20)')
#show()
#imshow(deblurImage,cmap=cm.gray); title('Deblur');
#show()
# PSNR of Unsharp Masking
psnrDeblur = findPSNR(deblurImage, onlyJoint)
print('PSNR of Gaussian Deblurring =', psnrDeblur)
#
# Sobel Filtering
sobelImage = 5 * skimage.filters.sobel(deblurImage)
sobelResult = np.add(deblurImage,sobelImage)
#imshow(deblurImage,cmap=cm.gray); title('Deblur');
#show()
#imshow(sobelImage,cmap=cm.gray); title('Sobel');
#show()
#imshow(sobelResult,cmap=cm.gray); title('Sobel Addition');
#show()
# PSNR of Sobel FIltering
psnrSobel = findPSNR(sobelResult, deblurImage)
print('PSNR of Sobel =', psnrSobel)
#
# Histogram Equalization
#neOnetoOneSobel = sobelResult/maxOfArray(sobelResult)
histImage = compInt * skimage.exposure.equalize_hist(sobelResult, 2**12)
#imshow(sobelResult,cmap=cm.gray); title('Sobel Deblur');
#show()
#imshow(histImage,cmap=cm.gray); title('HistEQ');
#show()
#hist(sobelResult); title('Sobel Hist');
#show()
#hist(histImage); title('EQed Hist');
#show()
# PSNR of Histogram Equalization
psnrHisteq = findPSNR(histImage, sobelResult)
print('PSNR of Histogram Equalization =', psnrHisteq)
#
# display orignial segment and filtered
imshow(onlyJoint, cmap = cm.gray); title('Orignial Joint')
show()
imshow(histImage, cmap = cm.gray); title('Filtered Joint')
show()
# PSNR
psnrFilteredOutput = findPSNR(histImage, onlyJoint)
print('PSNR of Filtered Output =', psnrFilteredOutput)
#
# Applying filters before matching
#deblur
gaussWhole = compInt * skimage.filters.gaussian(pixelsRaw,20)
wholeDeblur = np.subtract(pixelsRaw, gaussWhole)
#sobel
sobelofALL = 5 * skimage.filters.sobel(wholeDeblur)
sobelWhole = np.add(wholeDeblur,sobelofALL)
# no histogram eq bc it will scew all the contrast - so try and apply segmentation to this
#
# Test image
pathTest
gaussTest = compIntTest * skimage.filters.gaussian(pixelTest,20)
testDeblur = np.subtract(pixelTest, gaussTest)

sobelofTest = 5 * skimage.filters.sobel(testDeblur)
testWhole = np.add(testDeblur,sobelofTest)
#
# Feature Detection and Matching
# SIFT - limitations , it needs an imput image with noticable contrast differences
# I feel that this might be useful if we apply our filters first to make edges and contrast differences 
# notable in the image and then doing image segmentation
#
# The example below covers an example of matching the sobel filtering output above and matchinng it with the final
# output of the image
describeExtract = skimage.feature.SIFT() # initiate describeExtract as a SIFT() detection class

describeExtract.detect_and_extract(histImage) # needs image with greater intensity contrast to find features
keyPoint1 = describeExtract.keypoints # creates key points for features that are detected
descriptor1 = describeExtract.descriptors # describes the key point based on relation to other key points and 
                                          # location

describeExtract.detect_and_extract(sobelResult)
keyPoint2 = describeExtract.keypoints
descriptor2 = describeExtract.descriptors

describeExtract.detect_and_extract(sobelWhole) # not applied yet
keyPoint3 = describeExtract.keypoints
descriptor3 = describeExtract.descriptors

describeExtract.detect_and_extract(testWhole) # not applied yet
keyPoint4 = describeExtract.keypoints
descriptor4 = describeExtract.descriptors

matches12 = skimage.feature.match_descriptors(descriptor1, descriptor2, max_ratio = 0.6, cross_check = True)
matches13 = skimage.feature.match_descriptors(descriptor1, descriptor3, max_ratio = 0.6, cross_check = True)
matches14 = skimage.feature.match_descriptors(descriptor1, descriptor4, max_ratio = 0.6, cross_check = True)
# mathes the descriptors with simalaritites for matching later on
#
# outputs an aray with inputs such as [m, n]
# m correlates to the key point in the first image and n correlates to the keypoint on the second image
# that is assumed to bet he same as this one
#
# max_ratio Maximum ratio of distances between first and second closest descriptor in the second set of 
# descriptors. This threshold is useful to filter ambiguous matches between the two descriptor sets.
#
#cross_check to verify that a key point matches with the other - that it is infact the best key point
# to match with this image
fig, ax = subplots(nrows = 1, ncols = 3, figsize = (11,8))
print(ax)

gray()

skimage.feature.plot_matches(ax[0], histImage, sobelResult, keyPoint1, keyPoint2, matches12[::40])
skimage.feature.plot_matches(ax[1], histImage, sobelWhole, keyPoint1, keyPoint3, matches13[::40])
skimage.feature.plot_matches(ax[2], histImage, testWhole, keyPoint1, keyPoint3, matches13[::40])
# Will display every 40 matches detected and maps them side by side
#
# for now looks cluttered but this will be more simpler when applying it to the whole radiograph rather than
# just the segment heree
ax[0].axis('off')
ax[0].set_title('Final w/ Sobel')
ax[1].axis('off')
ax[1].set_title('Final w/ Sobel All Radioograph')
ax[2].axis('off')
ax[2].set_title('Final w/ Test Left')
show()
# from this output we can try and crop out the keypoints at boundaries to get a segmented image of the regions
# of interest w/o manualintervention
# we would use these key points form this radiograph to match the features with another radiograph exhibiting
# same view, may be tricky to get to work but may just do the trick
#
# will work on seeing if this works for other radiograph images showing the same view
#
# Seems to work well for now on both left and right extremities
# the test image is R2/L_AP_Elbow_6