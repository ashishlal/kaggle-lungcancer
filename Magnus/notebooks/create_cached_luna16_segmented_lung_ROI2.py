
# coding: utf-8

# In[1]:

import numpy as np
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
from glob import glob
from random import randint, choice
try:
    from tqdm import tqdm # long waits are not fun
except:
    print('TQDM does make much nicer wait bars...')
    tqdm = lambda x: x
import datetime


# In[2]:

working_path = "/home/watts/lal/Kaggle/lung_cancer/cache/luna16/512_512_16/"
file_list=glob(working_path+"images_*.npy")


# In[3]:

num_slices = 16
img_width = 128
img_height = 128


# In[4]:

for fcount, img_file in enumerate(tqdm(file_list)):
#     continue
#     if fcount != 0:
#         continue
    # I ran into an error when using Kmean on np.float16, so I'm using np.float64 here
    imgs_to_process = np.load(img_file).astype(np.float64) 
    # print "on image", img_file
    for i in range(len(imgs_to_process)):
        img = imgs_to_process[i]
        #Standardize the pixel values
        mean = np.mean(img)
        std = np.std(img)
        img = img-mean
        img = img/std
        # Find the average pixel value near the lungs
        # to renormalize washed out images
        middle = img[100:400,100:400] 
        mean = np.mean(middle)  
        max = np.max(img)
        min = np.min(img)
        # To improve threshold finding, I'm moving the 
        # underflow and overflow on the pixel spectrum
        img[img==max]=mean
        img[img==min]=mean
        #
        # Using Kmeans to separate foreground (radio-opaque tissue)
        # and background (radio transparent tissue ie lungs)
        # Doing this only on the center of the image to avoid 
        # the non-tissue parts of the image as much as possible
        #
        kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
        centers = sorted(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centers)
        thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image
        #
        # I found an initial erosion helful for removing graininess from some of the regions
        # and then large dialation is used to make the lung region 
        # engulf the vessels and incursions into the lung cavity by 
        # radio opaque tissue
        #
        eroded = morphology.erosion(thresh_img,np.ones([4,4]))
        dilation = morphology.dilation(eroded,np.ones([10,10]))
        #
        #  Label each region and obtain the region properties
        #  The background region is removed by removing regions 
        #  with a bbox that is to large in either dimnsion
        #  Also, the lungs are generally far away from the top 
        #  and bottom of the image, so any regions that are too
        #  close to the top and bottom are removed
        #  This does not produce a perfect segmentation of the lungs
        #  from the image, but it is surprisingly good considering its
        #  simplicity. 
        #
        labels = measure.label(dilation)
        label_vals = np.unique(labels)
        regions = measure.regionprops(labels)
        good_labels = []
        for prop in regions:
            B = prop.bbox
            if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
                good_labels.append(prop.label)
        mask = np.ndarray([512,512],dtype=np.int8)
        #mask = np.ndarray([41,41],dtype=np.int8)
        mask[:] = 0
        #
        #  The mask here is the mask for the lungs--not the nodes
        #  After just the lungs are left, we do another large dilation
        #  in order to fill in and out the lung mask 
        #
        for N in good_labels:
            mask = mask + np.where(labels==N,1,0)
        mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
        imgs_to_process[i] = mask
    np.save(img_file.replace("images","lungmask"),imgs_to_process)
    


# In[5]:


#
#    Here we're applying the masks and cropping and resizing the image
#

# num_slices = 16

import traceback
import warnings

#warnings.simplefilter("error")

file_list=glob(working_path+"lungmask_*.npy")
out_images = []      #final set of images
out_nodemasks = []   #final set of nodemasks
num_empty_slices = 0
num_images = 0
try:
    for fcount, fname in enumerate(tqdm(file_list)):
#         if fcount != 199:
#             continue
        #print "working on file ", fname
        imgs_to_process = np.load(fname.replace("lungmask","images"))
        masks = np.load(fname)
        node_masks = np.load(fname.replace("lungmask","masks"))
        my_out_images = []
        my_out_nodemasks = []
        my_pad_img = imgs_to_process[0]
        my_pad_nodemask = node_masks[0]
        #print imgs_to_process.shape
        for i in range(len(imgs_to_process)):
            mask = masks[i]
            node_mask = node_masks[i]
            img = imgs_to_process[i]
            new_size = [512,512]   # we're scaling back up to the original size of the image
            # new_size = [40,40]   
            img= mask*img          # apply lung mask
            empty_slice = not img[mask>0].any()
            if empty_slice == True:
                num_empty_slices = num_empty_slices + 1
            #    continue
            #
            # renormalizing the masked image (in the mask region)
            #
            nz = np.count_nonzero(img[mask >0])
            if nz == 0:
                continue
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
            #min_row = 40
            max_row = 0
            min_col = 512
            #min_col = 40
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
            if max_row-min_row <5 or max_col-min_col<5:  # skipping all images with no good regions
                pass
            else:
                # moving range to -1 to 1 to accomodate the resize function
                nz = np.count_nonzero(img)
                if nz == 0:
                    continue
                mean = np.mean(img)
                img = img - mean
                nz = np.count_nonzero(img)
                if nz == 0:
                    continue
                min = np.min(img)
                max = np.max(img)
                img = img/(max-min)
                new_img = resize(img,[512,512])
                new_node_mask = resize(node_mask[min_row:max_row,min_col:max_col],[512,512])
                nz1 = np.count_nonzero(node_mask)
                nz2 = np.count_nonzero(new_node_mask)
                if nz1 == 0 or nz2 == 0:
#                     print fcount, i
#                     print nz1, nz2
#                     print min_row, max_row, min_col, max_col
                    continue
                my_pad_img = new_img
                my_pad_nodemask = new_node_mask
                out_images.append(new_img)
                out_nodemasks.append(new_node_mask)
                my_out_images.append(new_img)
                my_out_nodemasks.append(new_node_mask)
        #print len(my_out_images)
        if len(my_out_images) == 0:
            continue
        while(len(my_out_images) < num_slices):
            index = randint(0, len(my_out_images)-1)
            my_pad_image = my_out_images[index]
            my_pad_nodemask = my_out_nodemasks[index]
            my_out_images.append(my_pad_img)
            my_out_nodemasks.append(my_pad_nodemask)
        #print len(my_out_images)
        my_fname = fname.replace("lungmask","my_images")
        my_nodemask_fname = fname.replace("lungmask","my_masks")
        # print 'writing %s, %s' %(my_fname, my_nodemask_fname)
        num_images += 1
        np.save(my_fname,my_out_images)
        np.save(my_nodemask_fname,my_out_nodemasks) 
except:
    print traceback.format_exc()
print num_empty_slices
print num_images


# In[ ]:

#
#  Writing out images and masks as 1 channel arrays for input into network
#
#final_images = np.ndarray([num_images,1,512,512],dtype=np.float32)
#final_masks = np.ndarray([num_images,1,512,512],dtype=np.float32)
#final_images = np.ndarray([num_images,1,40,40],dtype=np.float32)
#final_masks = np.ndarray([num_images,1,40,40],dtype=np.float32)
# for i in range(num_images):
#     final_images[i,0] = out_images[i]
#     final_masks[i,0] = out_nodemasks[i]

# rand_i = np.random.choice(range(num_images),size=num_images,replace=False)
# test_i = int(0.2*num_images)
# np.save(working_path+"trainImages_40_40.npy",final_images[rand_i[test_i:]])
# np.save(working_path+"trainMasks_40_40.npy",final_masks[rand_i[test_i:]])
# np.save(working_path+"testImages_40_40.npy",final_images[rand_i[:test_i]])
# np.save(working_path+"testMasks_40_40.npy",final_masks[rand_i[:test_i]])


# In[23]:

# np.save(working_path+"myTrainImages_40_40.npy",final_images)
# np.save(working_path+"myTrainMasks_40_40.npy",final_masks)


# In[16]:

num_images = 1155
file_list=glob(working_path+"my_masks_*.npy")
total_masks = 0
num_zeros = 0
my_final_masks = np.ndarray([num_images,1,num_slices,img_width,img_height],dtype=np.float32)
for i, mask_file in enumerate(tqdm(file_list)):
    zero_mask = 0
    masks = np.load(mask_file)
    masks = resize(masks,[num_slices,img_width,img_height])
    if (num_slices, img_height, img_width) != masks.shape:
        print i
        print masks.shape
    #print final_images[i,0].shape
    
    for j in range(num_slices):
        mask = masks[j]
        nz = np.count_nonzero(mask)
        if nz == 0:
            print 'mask %d in slice %s is all zero..' % (j, mask_file)
            zero_mask = 1
            continue
    if zero_mask == 1:
        continue
        
    nz = np.count_nonzero(masks)
    if nz == 0:
        print 'mask file %s is all 0' % mask_file
        num_zeros = num_zeros + 1
        continue
    my_final_masks[i,0] = masks
    total_masks = total_masks + 1
print('Total masks: %d, num zeros: %d' %(total_masks, num_zeros))
    
my_fname = working_path+'my_t_mask_%d_%d_%d.npy' % (num_slices, img_width, img_height)
print 'writing %s' % my_fname
np.save(my_fname,my_final_masks)
print 'Done'
now = datetime.datetime.now()
print now


# In[17]:

num_images = 1155
total_images = 0
num_zeros = 0

my_final_images = np.ndarray([num_images,1,num_slices,img_height,img_width],dtype=np.float32)
file_list=glob(working_path+"my_images_*.npy")
for i, img_file in enumerate(tqdm(file_list)):
    zero_image = 0
    #print img_file
    imgs = np.load(img_file)
    imgs = resize(imgs,[num_slices, img_width,img_height])
    if (num_slices, img_height, img_width) != imgs.shape:
        print i
        print imgs.shape
    #print final_images[i,0].shape
    for j in range(num_slices):
        img = imgs[j]
        nz = np.count_nonzero(img)
        if nz == 0:
            print 'image %d in slice %s is all zero..' % (i, img_file)
            zero_image = 1
            continue
    if zero_image == 1:
        continue
    nz = np.count_nonzero(imgs)
    if nz == 0:
        print 'slice %s is all zero..' % img_file
        num_zeros = num_zeros + 1
        continue
    my_final_images[i,0] = imgs
    total_images = total_images + 1

print('Total images: %d, num zeros: %d' %(total_images, num_zeros))
my_fname = working_path+'my_t_image_%d_%d_%d.npy' % (num_slices, img_width, img_height)
print 'writing %s' % my_fname
np.save(my_fname,my_final_images)
print 'Done'
now = datetime.datetime.now()
print now


# In[18]:

my_final_images = np.load(my_fname)
print my_final_images.shape


# In[19]:

num_zero = 0
for i in range(my_final_images.shape[0]):
    #for j in range(my_final_images.shape[2]):
    img = my_final_images[i,0]
    if (num_slices, img_height, img_width) != img.shape:
        print i
        print img.shape
    nz = np.count_nonzero(img)
    if nz == 0:
        num_zero += 1
print num_zero


# In[20]:

rand_i = np.random.choice(range(num_images),size=num_images,replace=False)
test_i = int(0.2*num_images)

my_fname_train = 'my_train_images_%d_%d_%d.npy' % (num_slices, img_width, img_height)
my_fname_test = 'my_test_images_%d_%d_%d.npy' % (num_slices, img_width, img_height)
print 'writing train and test images %s and %s..' % (my_fname_train, my_fname_test)
np.save(working_path+my_fname_train,my_final_images[rand_i[test_i:]])
np.save(working_path+my_fname_test,my_final_images[rand_i[:test_i]])
print 'Done'
now = datetime.datetime.now()
print now


# In[21]:

my_fname = working_path+'my_t_mask_%d_%d_%d.npy' % (num_slices, img_width, img_height)
my_final_masks = np.load(my_fname)
print my_final_masks.shape


# In[22]:

num_zero = 0
for i in range(my_final_masks.shape[0]):
    #for j in range(my_final_masks.shape[2]):
    img = my_final_masks[i,0]
    nz = np.count_nonzero(img)
    if nz == 0:
    #    print 'slice is 0...'
        num_zero += 1
print num_zero


# In[23]:

import datetime


# In[24]:

my_fname_train = 'my_train_masks_%d_%d_%d.npy' % (num_slices, img_width, img_height)
my_fname_test = 'my_test_masks_%d_%d_%d.npy' % (num_slices, img_width, img_height)  
print 'writing train and test masks %s and %s..' % (my_fname_train, my_fname_test)
np.save(working_path+my_fname_train,my_final_masks[rand_i[test_i:]])
np.save(working_path+my_fname_test,my_final_masks[rand_i[:test_i]])
print 'Done'
now = datetime.datetime.now()
print now


# In[25]:

my_fname = working_path+'my_t_image_%d_%d_%d.npy' % (num_slices, img_width, img_height)
my_final_images = np.load(my_fname)
print my_final_images.shape


# In[21]:

# num_image = 0
# my_out_image = np.ndarray([(num_images* num_slices),1,img_width,img_height],dtype=np.float32)
# for i in tqdm(range(my_final_images.shape[0])):
#     imgs = my_final_images[i,0]
#     assert(imgs.shape[0] == num_slices)
#     for j in range(imgs.shape[0]):
#         img = imgs[j]
#         my_fname = 'my_image_%d_%d_%d' % (num_image, img_width, img_height)
#         np.save(working_path+my_fname, img)
#         my_out_image[num_image,0] = img
#         num_image += 1
# print num_image
# assert(num_image == num_images * num_slices)
# my_fname = 'my_train_image_%d_%d' % (img_width, img_height)
# np.save(working_path+my_fname, my_out_image)
# print 'Done'
# now = datetime.datetime.now()
# print now


# In[22]:

# my_fname = working_path+'my_t_mask_%d_%d_%d.npy' % (num_slices, img_width, img_height)
# my_final_masks = np.load(my_fname)
# print my_final_masks.shape


# In[23]:

# num_mask = 0
# my_out_mask = np.ndarray([(num_images* num_slices),1,img_width,img_height],dtype=np.float32)
# for i in tqdm(range(my_final_masks.shape[0])):
#     masks = my_final_masks[i,0]
#     assert(masks.shape[0] == num_slices)
#     for j in range(masks.shape[0]):
#         mask = masks[j]
#         my_fname = 'my_mask_%d_%d_%d' % (num_mask, img_width, img_height)
#         np.save(working_path+my_fname, mask)
#         my_out_mask[num_mask,0] = mask
#         num_mask += 1
# print num_mask
# assert(num_mask == total_masks * num_slices)
# my_fname = 'my_train_mask_%d_%d' % (img_width, img_height)
# np.save(working_path+my_fname, my_out_mask)
# print 'Done'
# now = datetime.datetime.now()
# print now


# In[24]:

# rand_i = np.random.choice(range(num_image),size=num_image,replace=False)
# test_i = int(0.2*num_image)

# my_fname_train = 'my_train_image_%d_%d.npy' % (img_width, img_height)
# my_fname_test = 'my_test_image_%d_%d.npy' % (img_width, img_height)
# print 'writing train and test images %s and %s..' % (my_fname_train, my_fname_test)
# np.save(working_path+my_fname_train,my_out_image[rand_i[test_i:]])
# np.save(working_path+my_fname_test,my_out_image[rand_i[:test_i]])
# print 'Done'
# now = datetime.datetime.now()
# print now


# In[25]:

# my_fname_train = 'my_train_mask_%d_%d.npy' % (img_width, img_height)
# my_fname_test = 'my_test_mask_%d_%d.npy' % (img_width, img_height)
# print 'writing train and test masks %s and %s..' % (my_fname_train, my_fname_test)
# np.save(working_path+my_fname_train,my_out_mask[rand_i[test_i:]])
# np.save(working_path+my_fname_test,my_out_mask[rand_i[:test_i]])
# print 'Done'
# now = datetime.datetime.now()
# print now


# In[ ]:



