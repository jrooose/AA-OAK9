# AAOAK9_FinalPrototype.py
# UC Davis BIM110C
# Spring 2023
# May 22, 2023 7:00 PM
# Group 5: Anisha Kumar, Jasmine Rose Muolic, Blanca Osorio, Marc Ramirez, Emely Rivera 
# Automattic Assessment of Osteoarthritis in Canine Elbows
# Protype V3.1
import pydicom as dicom
from pydicom.errors import InvalidDicomError
from pydicom.uid import generate_uid, ExplicitVRLittleEndian, CTImageStorage
from pydicom.dataset import Dataset, FileMetaDataset
import datetime
import numpy as np
import math
import skimage
#from matplotlib.pyplot import figure, show, imshow, cm, axis
from pydicom.encoders import gdcm, pylibjpeg
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
def findPSNR(array1,array2,maxP = 2**12): # PSNR verified with skimage.metrics.peak_signal_noise_ratio() - - May 22, 2023 7:05 PM
    mse = np.mean((array1-array2)**2) # MSE verified with skimage.metrics.mean_squared_error() - - May 22, 2023 7:05 PM
    if mse == 0:
        return 100
    pSNR = 20 * math.log10(maxP/math.sqrt(mse))
    return pSNR
def maxOfArray(array):
    m, n = np.shape(array) # maxOfArray verified using np.max() May 20, 2023 6:00 PM
    tempMax = array[0:m,0]
    indexNum = 0
    for i in array:
        tempMax[indexNum] = max(i)
        indexNum +=1
    compInt = max(tempMax)
    return compInt
def origImageAndApplyFilter(path):
    raw = dicom.dcmread(path)
    pixel = raw.pixel_array
    compInt = maxOfArray(pixel)
    gaussImage = compInt * skimage.filters.gaussian(pixel,20) # standard deviation of 20
    deblurImage = np.subtract(pixel, gaussImage) # subtract the arrays
    sobelImage = 5 * skimage.filters.sobel(deblurImage)
    sobelResult = np.add(deblurImage,sobelImage)
    maxInt = maxOfArray(sobelResult)
    filtered = intesityThresh(sobelResult,r1 = 2**12-1,r2 = maxInt, setVal = 2**12-1)
    return raw, pixel, filtered
def convertFilteredToUint16(filt):
    gnew = filt//1
    genw1 = gnew.astype(np.uint16)
    #minvalUE = genw1.min()
    #newFilt1 = intesityThresh(genw1,0,r1=minvalUE,setVal = 0)
    fixedFilt = intesityThresh(genw1,2**16-1,r1=10000,setVal = 0)
    return fixedFilt
# Tests
def testMaxFunction(array): # maxOfArray verified using np.max() May 20, 2023 6:00 PM
    testMax = maxOfArray(array)
    npMax = np.max(array)
    if testMax == npMax:
        print('Results of testing maxOfArray() - - Passed')
    else:
        print('Results of testing maxOfArray() - - Failed')
    return
def reproducibilityTest(path): 
    out1 = origImageAndApplyFilter(path)
    out2 = origImageAndApplyFilter(path)
    psnr = skimage.metrics.peak_signal_noise_ratio(out1,out2,data_range=2**12)
    if psnr == 1:
        print('Results of testing reproducibilityTest() - - Passed')
    else:
        print('Results of testing reproducibilityTest() - - Failed')
    return
#
path = input('Enter path of the dicom file. Ensure file selected ends in ".dcm": ')
#path = '/Users/marcramirez/Desktop/AAOAK9/AnonymousRadPt2/R1/Elbow_Lf_Rf-145421/Right_Fore_CR_CD_Extremity_10/IM-0003-0001.dcm'
while True:# check if input file is dicom, later will include section outputing the radiograph to check if this
            # is the radiograph the user wants and if their is somethign wrong with the code as well
    try:
        x = dicom.dcmread(path) # Verified Monday 22, 2023 8:12 PM
        break
    except InvalidDicomError:
        path = input('The file you entered is not a DICOM cmopliant file. Please re-enter a path leading to a DICOM compliant file. ')
    except FileNotFoundError:
        path = input('There is no such file or directory:',path,'Please re-enter a path leading to a DICOM file. ')
rawDICOM, orig, filt = origImageAndApplyFilter(path)
#
ds = Dataset()
file_meta = FileMetaDataset()
newArray = convertFilteredToUint16(filt) # use this a bottom one only for filtered outputs
                                             # to set 10000 to 2**16-1 values  = 0  -- insignificant pixels, happens where 0's are
m,n = np.shape(newArray)
ds.PatientName = 'Filtered Version' ############################### get it from the orignial data file
ds.PatientID = 'AAOAK9_Test' ############################### Change to ask for patient ID
ds.StudyInstanceUID = generate_uid() # instances require new everytime
ds.StudyDescription = 'AAOAK9_FilteringTestImage' # copy from path oringial and state if ML or MLF
# inistead have this top tihing be the view so if 'Elbow LF RF'
# # change study desctiption beofre saving
dt = datetime.datetime.now()
ds.StudyDate = dt.strftime('%Y%m%d')
timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
ds.StudyTime = timeStr
ds.SeriesInstanceUID = generate_uid() # must be new
ds.SeriesNumber = '1'
ds.Modality = 'CR'
file_meta.MediaStorageSOPClassUID = CTImageStorage
ds.SOPClassUID = CTImageStorage
file_meta.MediaStorageSOPInstanceUID = generate_uid() # must be new
ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
file_meta.ImplementationClassUID = '1.2.826.0.1.3680043.8.498.26727104527805223125411960161162758320' # must be consistent in all radiographs we make
                                              # to identify DICOM application
file_meta.ImplementationVersionName = 'AAOAK9'
file_meta.PrivateInformationCreatorUID = '1.2.826.0.1.3680043.8.498.79620672134937555004390717653306568834' # must be consistent in all radiographs we make
                                                    # to identify me
# Image Module
ds.PatientOrientation = ['L', 'F'] ######################### get from orgnials
ds.Laterality = 'R'
ds.SamplesPerPixel = 1
ds.PhotometricInterpretation = 'MONOCHROME2'
ds.SpatialResolution = '0.125'
ds.ImagerPixelSpacing = [0.125, 0.125]
ds.Rows = m ############################
ds.Columns = n ############################
ds.ViewPosition = ''
ds.WindowCenter = str(float(n * 2))
ds.WindowWidth = str(float(n * 4))
ds.BitsAllocated = newArray.dtype.itemsize * 8
min_value = newArray.min()
max_value = newArray.max()
ds.BitsStored = int(np.ceil(np.log2(max_value - min_value + 1)))
ds.HighBit = ds.BitsStored - 1
ds.PixelRepresentation = 0
#
ds.PixelData = newArray.tobytes()
ds.file_meta = file_meta
ds.is_little_endian = True # needs to be true to show up
ds.is_implicit_VR = False # needs to be false to show up
newPath = path[:-4] + 'Filtered.dcm'
ds.save_as(newPath, write_like_original='False')