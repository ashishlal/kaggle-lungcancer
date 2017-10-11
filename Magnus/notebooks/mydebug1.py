
# coding: utf-8

# In[63]:

cd /home/watts/lal/Kaggle/lung_cancer


# In[64]:

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage, os
import argparse
from time import strftime
from tqdm import tqdm
import sys

from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize

from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import ndimage as ndi
from numpy.lib.format import open_memmap
import dicom
import scipy.misc
from utils.my_preprocessing import get_segmented_lungs, get_region_of_interest, remove_two_largest_connected
from utils.my_luna16_segment_lung_ROI import do_lungmask, do_final_processing, do_thresholding


# In[65]:

def get_current_date():
    return strftime('%Y%m%d')


# In[72]:

# Load the scans in given folder path
def read_ct_scan_gzuidhof(folder_path):
    #print folder_path
    slices = [dicom.read_file(folder_path + '/' + s) for s in os.listdir(folder_path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def get_pixels(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    return np.array(image, dtype=np.int16)

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

def plot_ct_scan(scan):
    f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(25, 25))
    for i in range(0, scan.shape[0], 5):
        plots[int(i / 20), int((i % 20) / 5)].axis('off')
        plots[int(i / 20), int((i % 20) / 5)].imshow(scan[i], cmap=plt.cm.bone) 

def segment_lung_from_ct_scan(ct_scan):
    return np.asarray([get_segmented_lungs(slice) for slice in ct_scan])


# In[67]:

args_size = 512
scan_folder = '0015ceb851d7251b8f399e39779d1e7d'
X_images_fname = 'cache/X_images_%s_%s_%s.npy' % (scan_folder, args_size, get_current_date())
X_lungmask_fname = 'cache/X_lungmask_%s_%s_%s.npy' % (scan_folder, args_size, get_current_date())
print X_images_fname


# In[68]:

import matplotlib.pyplot as plt
df = pd.read_csv('data/stage1_labels.csv')

df['scan_folder'] = df['id']
i = 0
row = df.iloc[i]
fname = os.path.join('data/stage1/', row['scan_folder'])
#try:
j = 2
scan_folder = row['scan_folder']
#print scan_folder
patient_slices = read_ct_scan_gzuidhof(fname)
patient_pixels = get_pixels(patient_slices)
#print len(patient_slices)

#img = patient_slices[j].pixel_array
img = patient_pixels[j]
print img.shape
print img.dtype
get_ipython().magic('matplotlib inline')

plt.imshow(img, cmap='gray')



# In[69]:

plot_ct_scan(patient_pixels)


# In[87]:

i = 0
row = df.iloc[i]
fname = os.path.join('data/stage1/', row['scan_folder'])

try:
    j = 0
    scan_folder = row['scan_folder']
    #print scan_folder
    print '0..'
    patient_slices = read_ct_scan_gzuidhof(fname)
    print '1..'
    patient_pixels = get_pixels(patient_slices)
    print patient_pixels.shape

    X_images_shape = (len(patient_slices), args_size, args_size)
    print X_images_shape
    X_images_fname = 'cache/X_images_%s_%s_%s.npy' % (scan_folder, args_size, get_current_date())
    #print X_images_shape, X_images_fname
    X_images_fp = np.memmap(X_images_fname, dtype=np.int64, mode='w+', shape=X_images_shape)

    X_lungmask_shape = (len(patient_slices), args_size, args_size)
    #print X_lungmask_shape
    X_lungmask_fname = 'cache/X_lungmask_%s_%s_%s.npy' % (scan_folder, args_size, get_current_date())
    print X_lungmask_shape, X_lungmask_fname
    #X_lungmask_fp = np.memmap(X_lungmask_fname, dtype=np.int64, mode='w+', shape=X_lungmask_shape)
    for slice in patient_slices:
        img = patient_pixels[j]
        X_images_fp[j] = img
        X_images_fp.flush()

        lungmask = do_lungmask(img)
        X_lungmask_fp[j] = lungmask
        X_lungmask_fp.flush()
        j = j+1
    print j

    X_images_fp = X_lungmask_fp*X_images_fp

    X_train_images_shape = (len(X_images_fp), args_size, args_size)
    print X_train_images_shape
    X_train_images_fname = 'cache/X_train_images_%s_%s_%s.npy' % (scan_folder, args_size, get_current_date())
    print X_train_images_fname
    #X_train_images_fp = np.memmap(X_train_images_fname, dtype=np.int64, mode='w+', shape=X_train_images_shape)

    print '0..'
    train_data = []
    for k in range(len(X_images_fp)):
        img = X_images_fp[k]
        mask = X_lungmask_fp[k]
        try:
            new_img = do_final_processing(img, mask)
            if new_img is not None:
                #X_train_images_fp[k] = new_img
                #X_train_images_fp.flush()
                #print '0a..'
                train_data.append(new_img)
                plt.imshow(new_img, cmap='gray')
        except:
            print 'failed in %s' % k
    print '1..'
    print len(train_data)
    print 'Deleting %s, %s ..' % (X_images_fname, X_lungmask_fname)
    print '2..'
    # now delete imgs and masks
    os.remove(X_images_fname) 
    print '3..'
    #os.remove(X_lungmask_fname) 
    print '4..'
except:
    print '%s has failed' % i
    #sys.exit(1)
print 'Done'


# In[91]:

images = np.stack([img for img in train_data])


# In[92]:

plt.imshow(images[0], cmap='gray')


# In[89]:

plt.imshow(train_data[0], cmap='gray')


# In[90]:

plt.imshow(train_data[1], cmap='gray')


# In[93]:

plot_ct_scan(images)


# In[ ]:




# In[52]:

X_images_shape = (len(patient_slices), args_size, args_size)
#print X_images_shape
X_images_fname = 'cache/X_images_%s_%s_%s.npy' % (scan_folder, args_size, get_current_date())
#print X_images_shape, X_images_fname
X_images_fp = np.memmap(X_images_fname, dtype=np.int64, mode='w+', shape=X_images_shape)

X_lungmask_shape = (len(patient_slices), args_size, args_size)
#print X_lungmask_shape
X_lungmask_fname = 'cache/X_lungmask_%s_%s_%s.npy' % (scan_folder, args_size, get_current_date())
print X_lungmask_shape, X_lungmask_fname
X_lungmask_fp = np.memmap(X_lungmask_fname, dtype=np.int64, mode='w+', shape=X_lungmask_shape)
#for slice in patient_slices:

#img = patient_slices[j].pixel_array
img = patient_pixels[j]
X_images_fp[j] = img
X_images_fp.flush()

lungmask = do_lungmask(img)
X_lungmask_fp[j] = lungmask
X_lungmask_fp.flush()
#j = j+1
#print j
#assert(j == len(patient_slices))
#label = row['cancer']
#y_fp[i] = label
#y_fp.flush()


# In[53]:

img = X_images_fp[j]
print img.shape
print img.dtype
get_ipython().magic('matplotlib inline')

plt.imshow(img, cmap='gray')


# In[54]:

thresh_img = do_thresholding(img)
plt.imshow(thresh_img, cmap='gray')


# In[55]:

eroded = morphology.erosion(thresh_img,np.ones([4,4]))
dilation = morphology.dilation(eroded,np.ones([10,10]))
labels = measure.label(dilation)
label_vals = np.unique(labels)
plt.imshow(labels)


# In[56]:

labels = measure.label(dilation)
label_vals = np.unique(labels)
regions = measure.regionprops(labels)
good_labels = []
for prop in regions:
    B = prop.bbox
    if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
        good_labels.append(prop.label)
mask = np.ndarray([512,512],dtype=np.int8)
mask[:] = 0
#
#  The mask here is the mask for the lungs--not the nodes
#  After just the lungs are left, we do another large dilation
#  in order to fill in and out the lung mask 
#
for N in good_labels:
    mask = mask + np.where(labels==N,1,0)
mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
plt.imshow(mask,cmap='gray')


# In[57]:

img = X_lungmask_fp[j]
print img.shape
print img.dtype
get_ipython().magic('matplotlib inline')

plt.imshow(img, cmap='gray')


# In[58]:

X_images_fp[j] = X_lungmask_fp[j] * X_images_fp[j]
img = X_images_fp[j]
print img.shape
print img.dtype
get_ipython().magic('matplotlib inline')

plt.imshow(img, cmap='gray')


# In[59]:

mask = X_lungmask_fp[j]
img= mask*img          # apply lung mask
#
# renormalizing the masked image (in the mask region)
#
new_mean = np.mean(img[mask>0])  
new_std = np.std(img[mask>0])
#
#  Pulling the background color up to the lower end
#  of the pixel range for the lungs
#
old_min = np.min(img)       # background color
img[img==old_min] = new_mean-1.2*new_std   # resetting backgound color
img = img-new_mean
img = img/new_std
#make image bounding box  (min row, min col, max row, max col)
labels = measure.label(mask)
regions = measure.regionprops(labels)
#
# Finding the global min and max row over all regions
#
min_row = 512
max_row = 0
min_col = 512
max_col = 0
for prop in regions:
    B = prop.bbox
    if min_row > B[0]:
        min_row = B[0]
    if min_col > B[1]:
        min_col = B[1]
    if max_row < B[2]:
        max_row = B[2]
    if max_col < B[3]:
        max_col = B[3]
width = max_col-min_col
height = max_row - min_row
if width > height:
    max_row=min_row+width
else:
    max_col = min_col+height
# 
# cropping the image down to the bounding box for all regions
# (there's probably an skimage command that can do this in one line)
# 
img = img[min_row:max_row,min_col:max_col]
mask =  mask[min_row:max_row,min_col:max_col]
new_img = ''
if max_row-min_row <5 or max_col-min_col<5:  # skipping all images with no god regions
    new_img = img
else:
    # moving range to -1 to 1 to accomodate the resize function
    mean = np.mean(img)
    img = img - mean
    min = np.min(img)
    max = np.max(img)
    img = img/(max-min)
    new_img = resize(img,[512,512])
plt.imshow(new_img, cmap='gray')


# In[60]:

new_img = do_final_processing(X_images_fp[j], X_lungmask_fp[j])

if new_img is not None:
    plt.imshow(new_img, cmap='gray')


# In[ ]:




# In[ ]:



X_train_images_shape = (len(X_images_fp), args_size, args_size)
#print X_train_images_shape
X_train_images_fname = 'cache/X_train_images_%s_%s_%s.npy' % (scan_folder, args_size, get_current_date())
#print X_train_images_fname
X_train_images_fp = np.memmap(X_train_images_fname, dtype=np.int64, mode='w+', shape=X_train_images_shape)


# In[ ]:

#print '0..'
for k in range(len(X_images_fp)):
    img = X_images_fp[k]
    mask = X_lungmask_fp[k]
    try:
        new_img = do_final_processing(img, mask)
        if new_img is not None:
            X_train_images_fp[k] = new_img
            X_train_images_fp.flush()
    except:
        print 'failed in %s' %k
print 'Deleting %s %s ..' % X_images_fname, X_lungmask_fname
# now delete imgs and masks
os.remove(X_images_fname) 
os.remove(X_lungmask_fname) 


# In[69]:

X_train_images_fname = 'X_train_images_0015ceb851d7251b8f399e39779d1e7d_512_20170215.npy'
imgs = np.memmap(X_train_images_fname, dtype=np.int64, mode='r', shape=(195,512,512))


# In[70]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.imshow(imgs[0])
#lt.show()


# In[ ]:



