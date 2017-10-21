
# coding: utf-8

# In[4]:

#cd /home/watts/lal/Kaggle/lung_cancer


# In[5]:

# From https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage, os
#os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cuda0,nvcc.fastmath=True,lib.cnmem=0.85'
import argparse
from time import strftime
from tqdm import tqdm
import sys
#import theano
from skimage.transform import resize
import datetime
#from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
#from skimage.measure import label,regionprops, perimeter
#from skimage.morphology import binary_dilation, binary_opening
#from skimage.filters import roberts, sobel
#from skimage import measure, feature
#from skimage.segmentation import clear_border
#from skimage import data
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import ndimage as ndi
from numpy.lib.format import open_memmap
import dicom
import scipy.misc
from utils.my_preprocessing import get_segmented_lungs, get_region_of_interest, remove_two_largest_connected
from utils.my_luna16_segment_lung_ROI import do_lungmask, do_final_processing


# In[6]:

# based on https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial

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


# In[7]:

def check_if_image_exists(fname):
    fname = os.path.join('data/stage1/stage1/', fname)
    return os.path.exists(fname)

def check_if_scan_exists(folder):
    folder = os.path.join('data/stage1/stage1/', folder)
    return os.path.isdir(folder)

def get_current_date():
    return strftime('%Y%m%d')


def load_images(df):
    for i, row in df['ImageFile'].iterrows():
        img = imread(row)
        yield img


# In[8]:

def read_ct_scan(folder_name):
    # Read the slices from the dicom file
    slices = [dicom.read_file(folder_name + filename) for filename in os.listdir(folder_name)]

    # Sort the dicom slices in their respective order
    slices.sort(key=lambda x: int(x.InstanceNumber))

    # Get the pixel values for all the slices
    slices = np.stack([s.pixel_array for s in slices])
    slices[slices == -2000] = 0
    return slices
def segment_lung_from_ct_scan(ct_scan):
    return np.asarray([get_segmented_lungs(slice) for slice in ct_scan])


# In[9]:

import math
import cv2

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]

# def chunks(l, n):
#     n = max(1, n)
#     return (l[i:i+n] for i in xrange(0, len(l), n))

def mean(l):
    return sum(l) / len(l)


# In[ ]:

IMG_PX_SIZE = 128
IMG_PX_SIZE_ORG = 512
HM_SLICES = 16


# In[10]:


new_slices = []

def get_new_slices(path):
    
    #print '01..'
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    #print len(slices)
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    #new_slices = []

    slices = [cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE_ORG,IMG_PX_SIZE_ORG))
              for each_slice in slices]
    
    chunk_sizes = int(math.ceil(len(slices) / HM_SLICES))
    #print '01..'
    #slices = [cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE_ORG,IMG_PX_SIZE_ORG)) 
    #                  for each_slice in patient_slices]

    #slices = patient_slices
    #slices[slices == -2000] = 0
    #print len(slices)
    #print '02..'
    #chunk_sizes = math.ceil(len(slices) / HM_SLICES)
    #chunk_sizes = len(slices) / HM_SLICES
    #print chunk_sizes
    #print '03..'
    
    for slice_chunk in chunks(slices, chunk_sizes):
        
        #print '031..'
        #print len(slice_chunk)
        #print len(slice_chunk)
        #print slice_chunk[0].shape
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        #print len(slice_chunk)
        #print slice_chunk[0].shape
        #print '0311..'
        new_slices.append(slice_chunk)

        #print '032..'
        if len(new_slices) == HM_SLICES-1:
            new_slices.append(new_slices[-1])

        #print '033..'
        if len(new_slices) == HM_SLICES-2:
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])

        #print '034..'
        if len(new_slices) == HM_SLICES+2:
            new_val = list(map(mean, zip(*[new_slices[HM_SLICES-1],new_slices[HM_SLICES],])))
            del new_slices[HM_SLICES]
            new_slices[HM_SLICES-1] = new_val

        #print '035..'
        if len(new_slices) == HM_SLICES+1:
            new_val = list(map(mean, zip(*[new_slices[HM_SLICES-1],new_slices[HM_SLICES],])))
            del new_slices[HM_SLICES]
            new_slices[HM_SLICES-1] = new_val
    return new_slices

def get_new_slices2(slices):
    
    slices = [each_slice.pixel_array for each_slice in slices]
    
    chunk_sizes = int(math.ceil(len(slices) / HM_SLICES))
    #print '01..'
    #slices = [cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE_ORG,IMG_PX_SIZE_ORG)) 
    #                  for each_slice in patient_slices]

    #slices = patient_slices
    #slices[slices == -2000] = 0
    #print len(slices)
    #print '02..'
    #chunk_sizes = math.ceil(len(slices) / HM_SLICES)
    #chunk_sizes = len(slices) / HM_SLICES
    #print chunk_sizes
    #print '03..'
    
    for slice_chunk in chunks(slices, chunk_sizes):
        
        #print '031..'
        #print len(slice_chunk)
        #print len(slice_chunk)
        #print slice_chunk[0].shape
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        #print len(slice_chunk)
        #print slice_chunk[0].shape
        #print '0311..'
        new_slices.append(slice_chunk)

        #print '032..'
        if len(new_slices) == HM_SLICES-1:
            new_slices.append(new_slices[-1])

        #print '033..'
        if len(new_slices) == HM_SLICES-2:
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])

        #print '034..'
        if len(new_slices) == HM_SLICES+2:
            new_val = list(map(mean, zip(*[new_slices[HM_SLICES-1],new_slices[HM_SLICES],])))
            del new_slices[HM_SLICES]
            new_slices[HM_SLICES-1] = new_val

        #print '035..'
        if len(new_slices) == HM_SLICES+1:
            new_val = list(map(mean, zip(*[new_slices[HM_SLICES-1],new_slices[HM_SLICES],])))
            del new_slices[HM_SLICES]
            new_slices[HM_SLICES-1] = new_val
    return new_slices

def get_new_slices3(slices):
    
    #slices = [each_slice.pixel_array for each_slice in slices]
    
    
    #print '01..'
    slices = [cv2.resize(np.array(each_slice),(IMG_PX_SIZE_ORG,IMG_PX_SIZE_ORG)) 
                      for each_slice in slices]
    
    chunk_sizes = int(math.ceil(len(slices) / HM_SLICES))

    #slices = patient_slices
    #slices[slices == -2000] = 0
    #print len(slices)
    #print '02..'
    #chunk_sizes = math.ceil(len(slices) / HM_SLICES)
    #chunk_sizes = len(slices) / HM_SLICES
    #print chunk_sizes
    #print '03..'
    
    for slice_chunk in chunks(slices, chunk_sizes):
        
        #print '031..'
        #print len(slice_chunk)
        #print len(slice_chunk)
        #print slice_chunk[0].shape
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        #print len(slice_chunk)
        #print slice_chunk[0].shape
        #print '0311..'
        new_slices.append(slice_chunk)

        #print '032..'
        if len(new_slices) == HM_SLICES-1:
            new_slices.append(new_slices[-1])

        #print '033..'
        if len(new_slices) == HM_SLICES-2:
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])

        #print '034..'
        if len(new_slices) == HM_SLICES+2:
            new_val = list(map(mean, zip(*[new_slices[HM_SLICES-1],new_slices[HM_SLICES],])))
            del new_slices[HM_SLICES]
            new_slices[HM_SLICES-1] = new_val

        #print '035..'
        if len(new_slices) == HM_SLICES+1:
            new_val = list(map(mean, zip(*[new_slices[HM_SLICES-1],new_slices[HM_SLICES],])))
            del new_slices[HM_SLICES]
            new_slices[HM_SLICES-1] = new_val
    return new_slices



# In[11]:



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', required=True, type=int, help='Original Size of the image')
    parser.add_argument('--lung_size', required=True, type=int, help='Size of the lung image')
    parser.add_argument('--nslices', required=True, type=int, help='Number of slices')
    parser.add_argument('--overwrite', action='store_true', help='Overwirte existing cache')

    args = parser.parse_args()

    IMG_PX_SIZE = args.lung_size
    IMG_PX_SIZE_ORG = args.size
    HM_SLICES = args.nslices

    df = pd.read_csv('data/stage1/stage1_labels.csv')

    df['scan_folder'] = df['id']

    df['exist'] = df['scan_folder'].apply(check_if_scan_exists)

    print '%i does not exists' % (len(df) - df['exist'].sum())
    print df[~df['exist']]

    df = df[df['exist']]
    df = df.reset_index(drop=True)
    
    y_fname = 'cache/y_%s.npy' % (get_current_date())
    y_shape = (len(df))

#     if os.path.exists(y_fname) and not args.overwrite:
#         print '%s exists. Use --overwrite' % y_fname
#         sys.exit(1)

   
    #y_fp = np.memmap(y_fname, dtype=np.int32, mode='w+', shape=y_shape)
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
#         continue
#         if i != 0:
#            continue
        fname = os.path.join('data/stage1/stage1/', row['scan_folder'])
        try:
            j = 0
            scan_folder = row['scan_folder']
            #print scan_folder
            patient_slices = read_ct_scan_gzuidhof(fname)
            patient_pixels = get_pixels(patient_slices)
            #print len(patient_slices)
            
            #X_images_shape = (len(patient_slices), args.size, args.size)
            #print X_images_shape
            #X_images_fname = 'cache/X_images_%s_%s_%s.npy' % (scan_folder, args.size, get_current_date())
            #print X_images_shape, X_images_fname
            #X_images_fp = np.memmap(X_images_fname, dtype=np.float32, mode='w+', shape=X_images_shape)
            X_images_fp = np.ndarray([HM_SLICES,IMG_PX_SIZE_ORG,IMG_PX_SIZE_ORG],dtype=np.float32)
            
            #X_lungmask_shape = (len(patient_slices), args.size, args.size)
            #print X_lungmask_shape
            #X_lungmask_fname = 'cache/X_lungmask_%s_%s_%s.npy' % (scan_folder, args.size, get_current_date())
            # print X_lungmask_shape, X_lungmask_fname
            #X_lungmask_fp = np.memmap(X_lungmask_fname, dtype=np.float32, mode='w+', shape=X_lungmask_shape)
            X_lungmask_fp = np.ndarray([HM_SLICES,IMG_PX_SIZE_ORG,IMG_PX_SIZE_ORG],dtype=np.float32)
            
            # each new_slice is an average of chunk_size slices
            new_slices = []
            #new_slices = get_new_slices(fname)
            #pix_resampled, my_spacing = resample(patient_pixels, patient_slices, [1,1,1])
            new_slices = get_new_slices2(patient_slices)
            #new_slices = get_new_slices3(pix_resampled)
            # print len(new_slices)
            
            for s in new_slices:
                #img = patient_pixels[j]
                #print '0..'
                #print len(s)
                img = np.reshape(s, (IMG_PX_SIZE_ORG, IMG_PX_SIZE_ORG))
                #print '0a..'
                img[img == -2000] = 0
                #print '0b..'
                #print img
                nz = np.count_nonzero(img)
                if nz == 0:
                    print 'slice all zero..........'
                #print '1..'
                X_images_fp[j] = img
                #X_images_fp.flush()
                #print '2..'
                lungmask = do_lungmask(img)
                #print '3..'
                X_lungmask_fp[j] = lungmask
                #X_lungmask_fp.flush()
                #print '4..'
                j = j+1
            #print j
            assert(j == len(new_slices))
            #label = row['cancer']
            #y_fp[i] = label
            #y_fp.flush()
            
            #X_images_fp = X_lungmask_fp*X_images_fp
            
            # X_segmented_images_shape = (len(X_images_fp), args.size, args.size)
            # print X_segmented_images_shape
            X_segmented_images_fname = 'cache/X_segmented_lungs_%s_%s_%s_%s.npy' % (scan_folder, HM_SLICES, IMG_PX_SIZE, IMG_PX_SIZE)
            #print X_segmented_images_fname
            #X_segmented_images_fp = np.memmap(X_train_images_fname, dtype=np.int64, mode='w+', shape=X_train_images_shape)
            
            #print '0..'
            #segmented_lungs = []
            segmented_lungs = np.ndarray([1,HM_SLICES,IMG_PX_SIZE,IMG_PX_SIZE],
                                         dtype=np.float32)
            # print len(new_slices)
            for k in range(len(new_slices)):
                img = X_images_fp[k]
                mask = X_lungmask_fp[k]
                #print img.shape
                #print mask.shape
                try:
                    #print '0a..'
                    new_img = do_final_processing(img, mask)
                    #print new_img.shape
                    #print '0b..'
                    if new_img is not None:
                        #X_segmented_images_fp[k] = new_img
                        #X_segmented_images_fp.flush()
                        #segmented_lungs.append(new_img)
                        #print '1..'
                        new_img = resize(new_img,[IMG_PX_SIZE,IMG_PX_SIZE])
                        #print new_img.shape
                        #print '2..'
                        #segmented_lungs.append(new_img)
                        segmented_lungs[0, k] = new_img
                        #print '3..'
                        
                except:
                    print 'failed in %s' %k
                    sys.exit(1)
            #segmented_lung_images = np.stack([img for img in segmented_lungs])
            np.save(X_segmented_images_fname, segmented_lungs) 
            #print 'Deleting %s %s ..' % (X_images_fname, X_lungmask_fname)
            # now delete imgs and masks
            #os.remove(X_images_fname) 
            #os.remove(X_lungmask_fname) 
        except:
            print '%s has failed' % i
            #sys.exit(1)
    # now read, used for 2D segmentation 
#     num_images = 1397
#     num_seg_slices = 0
#     num_slices = HM_SLICES
#     img_width = args.lung_size
#     img_height = args.lung_size
#     my_out_seg_lung = np.ndarray([(num_images* num_slices),1,img_width,img_height],dtype=np.float32)
#     for i, row in tqdm(df.iterrows(), total=len(df)):
#         scan_folder = row['scan_folder']
#         X_segmented_lungs_fname = 'cache/X_segmented_lungs_%s_%s_%s_%s.npy' % (scan_folder, HM_SLICES, IMG_PX_SIZE, IMG_PX_SIZE)
#         seg_lung = np.load(X_segmented_lungs_fname)
#         assert(seg_lung.shape[1] == HM_SLICES)
#         for j in range(seg_lung.shape[1]):
#             lung_slice = seg_lung[0,j]
#             assert(lung_slice.shape[0] == img_width)
#             my_fname = 'cache/X_segmented_lung_%s_%d_%d_%d' % (scan_folder,j,img_width, img_height)
#             np.save(my_fname, lung_slice)
#             my_out_seg_lung[num_seg_slices,0] = lung_slice
#             num_seg_slices += 1
#     print num_seg_slices
#     assert(num_seg_slices == num_images * num_slices)
print 'Done'
now = datetime.datetime.now()
print now


# In[ ]:



