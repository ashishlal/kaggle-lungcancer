
# coding: utf-8

# In[ ]:

#cd /home/watts/lal/Kaggle/lung_cancer


# In[ ]:

# based on https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
get_ipython().magic('matplotlib inline')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Some constants 
INPUT_FOLDER = '../data/stage1/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()


# # Loading the files

# Dicom is the de-facto file standard in medical imaging. This is my first time working with it, but it seems to be fairly straight-forward. 
# 
# These files contain a lot of metadata (such as the pixel size, so how long one pixel is in every dimension in the real world).
# 
# This pixel size/coarseness of the scan differs from scan to scan (e.g. the distance between slices may differ), which can hurt performance of CNN approaches. 
# 
# We can deal with this by isomorphic resampling, which we will do later.
# 
# Below is code to load a scan, which consists of multiple slices, which we simply save in a Python list. 
# 
# Every folder in the dataset is one scan (so one patient). One metadata field is missing, the pixel size in the Z direction, which is the slice thickness. 
# 
# Fortunately we can infer this, and we add this to the metadata.

# In[ ]:

# Load the scans in given folder path
def load_scan(folder_path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(folder_path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices


# The unit of measurement in CT scans is the Hounsfield Unit (HU), which is a measure of radiodensity. CT scanners are carefully calibrated to accurately measure this. From Wikipedia:
# HU examples

# In[2]:

get_ipython().system('pwd')


# <img src="../data/Wikipedia-CT-scan-substance.png"/>

# By default however, the returned values are not in this unit. Let's fix this.
# 
# Some scanners have cylindrical scanning bounds, but the output image is square. The pixels that fall outside of these bounds get the fixed value -2000. The first step is setting these values to 0, which currently corresponds to air. Next, let's go back to HU units, by multiplying with the rescale slope and adding the intercept (which are conveniently stored in the metadata of the scans!).

# In[ ]:

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)


# # Resampling

# A scan may have a pixel spacing of [2.5, 0.5, 0.5], which means that the distance between slices is 2.5 millimeters. For a different scan this may be [1.5, 0.725, 0.725], this can be problematic for automatic analysis (e.g. using ConvNets)!
# A common method of dealing with this is resampling the full dataset to a certain isotropic resolution. If we choose to resample everything to 1mm1mm1mm pixels we can use 3D convnets without worrying about learning zoom/slice thickness invariance.
# Whilst this may seem like a very simple step, it has quite some edge cases due to rounding. Also, it takes quite a while.
# Below code worked well for us (and deals with the edge cases):

# In[ ]:

def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing


# In[ ]:

pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1,1,1])
print("Shape before resampling\t", first_patient_pixels.shape)
print("Shape after resampling\t", pix_resampled.shape)


# # Lung segmentation

# In order to reduce the problem space, we can segment the lungs (and usually some tissue around it). The method that me and my student colleagues developed was quite effective.
# It involves quite a few smart steps. It consists of a series of applications of region growing and morphological operations. In this case, we will use only connected component analysis.
# 
# The steps:
# 1. Threshold the image (-320 HU is a good threshold, but it doesn't matter much for this approach)
# 2. Do connected components, determine label of air around person, fill this with 1s in the binary image
# 3. Optionally: For every axial slice in the scan, determine the largest solid connected component (the body+air around the person), and set others to 0. This fills the structures in the lungs in the mask.
# 4. Keep only the largest air pocket (the human body has other pockets of air here and there).

# In[ ]:



