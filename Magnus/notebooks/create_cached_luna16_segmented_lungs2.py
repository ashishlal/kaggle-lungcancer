
# coding: utf-8

# In[1]:

#cd /home/watts/lal/Kaggle/lung_cancer


# In[1]:

from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd
import os
from random import randint, choice
try:
    from tqdm import tqdm # long waits are not fun
except:
    print('TQDM does make much nicer wait bars...')
    tqdm = lambda x: x


# In[2]:

luna_path = "/home/watts/lal/Kaggle/lung_cancer/data_luna16/"
luna_subset_path = luna_path+"subsetall_my/"
output_path = "/home/watts/lal/Kaggle/lung_cancer/cache/luna16/512_512_16/"
file_list=glob(luna_subset_path+"*.mhd")


# In[3]:

def make_mask(center,diam,z,width,height,spacing,origin):
    '''
Center : centers of circles px -- list of coordinates x,y,z
diam : diameters of circles px -- diameter
widthXheight : pixel dim of image
spacing = mm/px conversion rate np array x,y,z
origin = x,y,z mm np.array
z = z position of slice in world coordinates mm
    '''
    print('1..')
    print(center,diam,z,width,height,spacing,origin)
    mask = np.zeros([height,width]) # 0's everywhere except nodule swapping x,y to match img
    #convert to nodule space from world coordinates

    # Defining the voxel range in which the nodule falls
    v_center = (center-origin)/spacing
    v_diam = int(diam/spacing[0]+5)
    v_xmin = np.max([0,int(v_center[0]-v_diam)-5])
    v_xmax = np.min([width-1,int(v_center[0]+v_diam)+5])
    v_ymin = np.max([0,int(v_center[1]-v_diam)-5]) 
    v_ymax = np.min([height-1,int(v_center[1]+v_diam)+5])

    print('2..')
    print(v_center,v_diam,v_xmin,v_xmax,v_ymin,v_ymax)
    
    v_xrange = range(v_xmin,v_xmax+1)
    v_yrange = range(v_ymin,v_ymax+1)

    print('3..')
    print(v_xrange, v_yrange)
    
    # Convert back to world coordinates for distance calculation
    x_data = [x*spacing[0]+origin[0] for x in range(width)]
    y_data = [x*spacing[1]+origin[1] for x in range(height)]

    #print('4..')
    #print(x_data, y_data)
    
    # Fill in 1 within sphere around nodule
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0]*v_x + origin[0]
            p_y = spacing[1]*v_y + origin[1]
            #print(p_x, p_y)
            if np.linalg.norm(center-np.array([p_x,p_y,z]))<=diam:
                mask[int((p_y-origin[1])/spacing[1]),int((p_x-origin[0])/spacing[0])] = 1.0
    return(mask)

def make_mask2(center,diam,z,width,height,spacing,origin):
    '''
Center : centers of circles px -- list of coordinates x,y,z
diam : diameters of circles px -- diameter
widthXheight : pixel dim of image
spacing = mm/px conversion rate np array x,y,z
origin = x,y,z mm np.array
z = z position of slice in world coordinates mm
    '''
    #print('1..')
    #print(center,diam,z,width,height,spacing,origin)
    mask = np.zeros([height,width]) # 0's everywhere except nodule swapping x,y to match img
    #convert to nodule space from world coordinates

    # Defining the voxel range in which the nodule falls
    v_center = (center-origin)/spacing
    v_diam = int(diam/spacing[0]+5)
    v_diam_z = int(diam/spacing[2]+5)
    v_xmin = np.max([0,int(v_center[0]-v_diam)-5])
    v_xmax = np.min([width-1,int(v_center[0]+v_diam)+5])
    v_ymin = np.max([0,int(v_center[1]-v_diam)-5]) 
    v_ymax = np.min([height-1,int(v_center[1]+v_diam)+5])
    v_zmax = np.max([0,int(v_center[2]-v_diam_z)+5]) 
    v_zmin = np.min([int(z-1),int(v_center[2]+v_diam_z)-5])
    
    #print('2..')
    #print(v_center,v_diam,v_diam_z,v_xmin,v_xmax,v_ymin,v_ymax,v_zmin,v_zmax)
    
    v_xrange = range(v_xmin,v_xmax+1)
    v_yrange = range(v_ymin,v_ymax+1)
    v_zrange = range(v_zmin,v_zmax+1)
    
    #print('3..')
    #print(v_xrange, v_yrange, v_zrange)
    
    # Convert back to world coordinates for distance calculation
    x_data = [x*spacing[0]+origin[0] for x in range(width)]
    y_data = [x*spacing[1]+origin[1] for x in range(height)]
    z_data = [x*spacing[2]+origin[2] for x in range(int(z))]
    
    #print('4..')
    #print(x_data, y_data)
    
    # Fill in 1 within sphere around nodule
    min_xy = 1000000
    min_xz = 1000000
    min_yz = 1000000
    min_xyz = 1000000
    for v_x in v_xrange:
        for v_y in v_yrange:
        #for v_z in v_zrange:
            p_x = spacing[0]*v_x + origin[0]
            p_y = spacing[1]*v_y + origin[1]
            #p_z = spacing[2]*v_z + origin[2]
            #print(p_x, p_y)
            dxy = np.linalg.norm(center-np.array([p_x,p_y,z]))
            #dxyz =np.linalg.norm(center-np.array([p_x,p_y,p_z]))
            if dxy < min_xy:
                min_xy = dxy
#             if dxyz < min_xyz:
#                 min_xyz = dxyz
            if (
                dxy <= diam+2
            ):
                mask[int((p_y-origin[1])/spacing[1]),int((p_x-origin[0])/spacing[0])] = 1.0
    #print('5..')
    #print(min_xy, diam+4)
    return(mask)


def matrix2int16(matrix):
    ''' 
matrix must be a numpy array NXN
Returns uint16 version
    '''
    m_min= np.min(matrix)
    m_max= np.max(matrix)
    matrix = matrix-m_min
    return(np.array(np.rint( (matrix-m_min)/float(m_max-m_min) * 65535.0),dtype=np.uint16))

#####################
#
# Helper function to get rows in data frame associated 
# with each file
def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return(f)


# In[4]:

#
# The locations of the nodes
df_node = pd.read_csv(luna_path+"annotations.csv")
df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
df_node = df_node.dropna()
df_node.head(5)


# In[5]:

zero_masks = [676, 17, 94, 102, 645, 799, 147, 248, 259, 141, 752, 799, 251, 870,
689,
17,
456,
235,
316,
442,
728,
300,
789,
389,
440,
181,
443,
444,
465,
541,
171,
749,
664,
84,
119,
812,
376,
401,
167,
799,
12,
373,
696,
91,
329,
770,
774,
]
zero_masks = sorted(set(zero_masks))


# In[6]:

print(zero_masks)


# In[9]:

num_slices = 16
my_i = 0
my_j = 0
num_images = 0
for fcount, img_file in enumerate(tqdm(file_list)):
#     if fcount != 443:
#         continue
#     if my_j == len(zero_masks):
#         break
#     if fcount != zero_masks[my_j]:
#         continue
#     print(my_j, fcount, len(zero_masks))
#     my_j += 1
    
#     if my_i == 1:
#         continue
    mini_df = df_node[df_node["file"]==img_file] #get all nodules associate with file
    if mini_df.shape[0]>0: # some files may not have a nodule--skipping those 
        # load the data once
        itk_img = sitk.ReadImage(img_file) 
        img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
        num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
        origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
        spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
        #print(img_array.shape)
        #print(origin)
        #print(spacing)
        # go through all nodes (why just the biggest?)
        #tot = 0
        for node_idx, cur_row in mini_df.iterrows(): 
#             if tot == 1:
#                 continue
#             my_i = 1
#             tot = 1
            node_x = cur_row["coordX"]
            node_y = cur_row["coordY"]
            node_z = cur_row["coordZ"]
            diam = cur_row["diameter_mm"]
            # just keep 3 slices
            # ashish: changed to 16
            imgs = np.ndarray([num_slices,height,width],dtype=np.float32)
            masks = np.ndarray([num_slices,height,width],dtype=np.uint8)
            center = np.array([node_x, node_y, node_z])   # nodule center
            center_org = center
            v_center = np.rint((center-origin)/spacing)  # nodule center in voxel space (still x,y,z ordering)
            #print(center)
            #print(v_center)
            total_images = 0
            succesful_z_index = []
            #for i, i_z in enumerate(np.arange(int(v_center[2])-1,
            #                 int(v_center[2])+(2 * num_slices-1)).clip(0, num_z-1)): # clip prevents going out of bounds in Z
            for i, i_z in enumerate(np.arange(int(v_center[2])-int(num_slices/2),
                             int(v_center[2])+int(num_slices/2)).clip(0, num_z-1)): # clip prevents going out of bounds in Z   
                #print(i,i_z)
                mask = make_mask2(center, diam, i_z*spacing[2]+origin[2],
                                 width, height, spacing, origin)
                attempts = 0
                nz = np.count_nonzero(mask)
                while nz == 0 and attempts < 5:
                    #print('skipping .. %d, %d, %d' %(i,i_z, attempts))
                    attempts += 1
                    # w in 10-25 range, h in 3.5
                    dx = randint(10,25)
                    dy = randint(10,25)
                    dz = randint(0,3)
                    random_shift = np.array([dx, dy, dz])
                    center = center_org - random_shift
                    mask = make_mask2(center, diam, i_z*spacing[2]+origin[2],
                                 width, height, spacing, origin)
                    nz = np.count_nonzero(mask)
                nz = np.count_nonzero(mask)
                if nz == 0:
                    continue
                nz = np.count_nonzero(img_array[i_z])
                if nz == 0:
                    continue
                #masks[i] = mask
                #imgs[i] = img_array[i_z]
                succesful_z_index.append(i_z)
                masks[total_images] = mask
                imgs[total_images] = img_array[i_z]
                total_images += 1
                if total_images >= num_slices:
                    break
            #print('got %d images from unique slices' % total_images)
            while (total_images < num_slices) and (len(succesful_z_index) > 0):
                i_z = choice(succesful_z_index)
                nz = 0
                attempts = 0
                while nz == 0 and attempts < 5:
                    attempts += 1
                    # w in 10-25 range, h in 3.5
                    dx = randint(10,25)
                    dy = randint(10,25)
                    dz = randint(0,3)
                    random_shift = np.array([dx, dy, dz])
                    center = center_org - random_shift
                    mask = make_mask2(center, diam, i_z*spacing[2]+origin[2],
                                 width, height, spacing, origin)
                    nz = np.count_nonzero(mask)
                nz = np.count_nonzero(mask)
                if nz == 0:
                    continue
                masks[total_images] = mask
                imgs[total_images] = img_array[i_z]
                total_images += 1
            if total_images < num_slices:
                continue
            num_images +=1
            #print('writing %d images to images_%04d_%04d.npy' % (total_images, fcount, node_idx))    
            np.save(os.path.join(output_path,"images_%04d_%04d.npy" % (fcount, node_idx)),imgs)
            np.save(os.path.join(output_path,"masks_%04d_%04d.npy" % (fcount, node_idx)),masks)
print(num_images)


# In[10]:

output_path1 = "/home/watts/lal/Kaggle/lung_cancer/cache/luna16/512_512_16/"
test = np.load(output_path1+"masks_0386_0511.npy")


# In[11]:

np.count_nonzero(test)


# In[12]:

print (test.shape)


# In[13]:

file_list=glob(output_path1+"images_*.npy")
num_zeros = 0
total = 0
for fcount, fname in enumerate(tqdm(file_list)):
    #print "working on file ", fname
    imgs = np.load(fname)
    nz = np.count_nonzero(imgs)
    if nz == 0:
        print('file %s is all zero..' % fname)
        num_zeros = num_zeros + 1
    total = total + 1
print('Total images %s, Num Zeros: %s' % (total, num_zeros))


# In[14]:

file_list=glob(output_path1+"images_*.npy")
num_zeros = 0
total = 0
for fcount, fname in enumerate(tqdm(file_list)):
    #if fcount != 0:
    #    continue
    #print "working on file ", fname
    imgs = np.load(fname)
    #print (imgs.shape)
    for i in range(imgs.shape[0]):
        img = imgs[i]
        nz = np.count_nonzero(img)
        if nz == 0:
            print('image %d in file %s is all zero' % (i, fname))
            num_zeros = num_zeros + 1
        total = total + 1
print('Total images %s, Num Zeros: %s' % (total, num_zeros))


# In[15]:

file_list=glob(output_path+"masks_*.npy")
num_zeros = 0
total = 0
for fcount, fname in enumerate(tqdm(file_list)):
    #print "working on file ", fname
    masks = np.load(fname)
    nz = np.count_nonzero(masks)
    if nz == 0:
        print('mask %s is all zero..' % fname)
        num_zeros = num_zeros + 1
    total = total + 1
print('Total masks %s, Num Zeros: %s' % (total, num_zeros))


# In[16]:

file_list=glob(output_path+"masks_*.npy")
num_zeros = 0
total = 0
for fcount, fname in enumerate(tqdm(file_list)):
    #print "working on file ", fname
    masks = np.load(fname)
    for i in range(num_slices):
        img = masks[i]
        nz = np.count_nonzero(img)
        if nz == 0:
            print('mask %d in file %s is zero' % (i, fname))
            num_zeros = num_zeros + 1
        total = total + 1
print('Total masks %s, Num Zeros: %s' % (total, num_zeros))


# In[ ]:



