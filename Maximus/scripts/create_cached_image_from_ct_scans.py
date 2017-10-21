
# coding: utf-8

# In[ ]:

# From https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage, os
import argparse
from time import strftime
from tqdm import tqdm
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
from utils.my_luna16_segment_lung_ROI import do_lungmask


# In[ ]:

# based on https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial

# Load the scans in given folder path
def read_ct_scan_gzuidhof(folder_path):
    print folder_path
    slices = [dicom.read_file(folder_path + '/' + s) for s in os.listdir(folder_path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

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


# In[ ]:

def check_if_image_exists(fname):
    fname = os.path.join('data/stage1/', fname)
    return os.path.exists(fname)

def check_if_scan_exists(folder):
    folder = os.path.join('data/stage1/', folder)
    return os.path.isdir(folder)

def get_current_date():
    return strftime('%Y%m%d')


def load_images(df):
    for i, row in df['ImageFile'].iterrows():
        img = imread(row)
        yield img


# In[ ]:

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


# In[ ]:

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', required=True, type=int, help='Size of the image')
    parser.add_argument('--overwrite', action='store_true', help='Overwirte existing cache')
    #parser.add_argument('--naive_shuai', action='store_true')

    args = parser.parse_args()

    df = pd.read_csv('data/stage1_labels.csv')

    df['scan_folder'] = df['id']

    df['exist'] = df['scan_folder'].apply(check_if_scan_exists)

    print '%i does not exists' % (len(df) - df['exist'].sum())
    print df[~df['exist']]

    df = df[df['exist']]
    df = df.reset_index(drop=True)
    
    X_fname = 'cache/X_%s_%s.npy' % (args.size, get_current_date())
    X_resampled_fname = 'cache/X_resampled_%s_%s.npy' % (args.size, get_current_date())
    y_fname = 'cache/y_%s.npy' % (get_current_date())
    X_spacing_fname = 'cache/X_%s.npy' % (get_current_date())
    
    #X_shape = (len(df), 3, args.size, args.size)
    y_shape = (len(df))
    X_spacing_shape = (len(df), 3)

    if os.path.exists(X_fname) and not args.overwrite:
        print '%s exists. Use --overwrite' % X_fname
        sys.exit(1)

    if os.path.exists(y_fname) and not args.overwrite:
        print '%s exists. Use --overwrite' % y_fname
        sys.exit(1)

    if os.path.exists(X_resampled_fname) and not args.overwrite:
        print '%s exists. Use --overwrite' % X_resampled_fname
        sys.exit(1)
        
    if os.path.exists(X_resampled_fname) and not args.overwrite:
        print '%s exists. Use --overwrite' % X_spacing_shape
        sys.exit(1)

    print 'Will write X to %s' % (X_fname)
    print 'Will write y to %s' % (y_fname)
    print 'Will write X_resampled to %s' % (X_resampled_fname)
    print 'Will write X_spacing to %s' % (X_spacing_fname)
    
    #X_fp = open_memmap(X_fname, mode='w+', dtype=np.int16, shape=(int(1E13),))
    #X_resampled_fp = open_memmap(X_resampled_fname, mode='w+', dtype=np.ubyte, shape=(int(1E13),))
    y_fp = np.memmap(y_fname, dtype=np.int32, mode='w+', shape=y_shape)
    #X_spacing_fp = np.memmap(X_spacing_fname, dtype=np.int32, mode='w+', shape=X_spacing_shape)
    
    much_data = []
    much_resampled_data = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        fname = os.path.join('data/stage1/', row['scan_folder'])

        #X_lungmask_fp = np.memmap(X_lungmask_fname, dtype=np.int32, mode='w+', shape=X_spacing_shape)
        try:
            patient_slices = read_ct_scan_gzuidhof(fname)
            for slice in patient_slices:
                img = slice.pixel_array
                lungmask = do_lungmask(img)
                print lungmask.shape
                print lungmask.dtype
                
            #patient_pixels = get_pixels_hu(patient_slices)
            label = row['cancer']
            
            #pix_resampled, spacing = resample(patient_pixels, patient_slices, [1,1,1])
            
            
            print patient_pixels.shape
            print patient_pixels.dtype
            print pix_resampled.shape
            print spacing
            print spacing.shape
            print spacing.dtype
            print patient_pixels.shape       
            #much_data.append(patient_pixels)
            
            y_fp[i] = label
            
            #much_resampled_data.append(pix_resampled)
            
            y_fp.flush()
            
        except:
            print '%s has failed' % i
            sys.exit(1)
    #np.save(X_fname, much_data)
    #np.save(X_resampled_fname, much_resampled_data)
    print 'Done'

