
# coding: utf-8

# In[ ]:

import numpy as np
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
from glob import glob
try:
    from tqdm import tqdm # long waits are not fun
except:
    print('TQDM does make much nicer wait bars...')
    tqdm = lambda x: x


# In[ ]:

# working_path = "/home/watts/lal/Kaggle/lung_cancer/cache/luna16/02242017_41_41_7/"
working_path = "/home/watts/lal/Kaggle/lung_cancer/cache/luna16/"
file_list=glob(working_path+"images_*.npy")


# In[ ]:

for fcount, img_file in enumerate(tqdm(file_list)):
    # I ran into an error when using Kmean on np.float16, so I'm using np.float64 here
    imgs_to_process = np.load(img_file).astype(np.float64) 
    #print "on image", img_file
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
    


# In[ ]:


#
#    Here we're applying the masks and cropping and resizing the image
#


file_list=glob(working_path+"lungmask_*.npy")
out_images = []      #final set of images
out_nodemasks = []   #final set of nodemasks
for fcount, fname in enumerate(tqdm(file_list)):
    #print "working on file ", fname
    imgs_to_process = np.load(fname.replace("lungmask","images"))
    masks = np.load(fname)
    node_masks = np.load(fname.replace("lungmask","masks"))
    my_out_images = []
    my_out_nodemasks = []
    my_pad_img = imgs_to_process[0]
    my_pad_nodemask = node_masks[0]
    for i in range(len(imgs_to_process)):
        mask = masks[i]
        node_mask = node_masks[i]
        img = imgs_to_process[i]
        new_size = [512,512]   # we're scaling back up to the original size of the image
        #new_size = [41,41]   
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
        #min_row = 41
        max_row = 0
        min_col = 512
        #min_col = 41
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
            
            mean = np.mean(img)
            img = img - mean
            min = np.min(img)
            max = np.max(img)
            img = img/(max-min)
            new_img = resize(img,[512,512])
            #new_img = resize(img,[41,41])
            new_node_mask = resize(node_mask[min_row:max_row,min_col:max_col],[512,512])
            #new_node_mask = resize(node_mask[min_row:max_row,min_col:max_col],[41,41])
            my_pad_img = new_img
            my_pad_nodemask = new_node_mask
            out_images.append(new_img)
            out_nodemasks.append(new_node_mask)
            my_out_images.append(new_img)
            my_out_nodemasks.append(new_node_mask)
    while(len(my_out_images) < 8):
        my_out_images.append(my_pad_img)
        my_out_nodemasks.append(my_pad_nodemask)
    my_fname = fname.replace("lungmask","my_images")
    my_nodemask_fname = fname.replace("lungmask","my_masks")
    #print 'writing %s, %s' %(my_fname, my_nodemask_fname)
    np.save(my_fname,my_out_images)
    np.save(my_nodemask_fname,my_out_nodemasks) 
num_images = len(out_images)
print num_images


# In[ ]:

#
#  Writing out images and masks as 1 channel arrays for input into network
#
final_images = np.ndarray([num_images,1,512,512],dtype=np.float32)
final_masks = np.ndarray([num_images,1,512,512],dtype=np.float32)
#final_images = np.ndarray([num_images,1,41,41],dtype=np.float32)
#final_masks = np.ndarray([num_images,1,41,41],dtype=np.float32)
for i in range(num_images):
    final_images[i,0] = out_images[i]
    final_masks[i,0] = out_nodemasks[i]

rand_i = np.random.choice(range(num_images),size=num_images,replace=False)
test_i = int(0.2*num_images)
np.save(working_path+"trainImages.npy",final_images[rand_i[test_i:]])
np.save(working_path+"trainMasks.npy",final_masks[rand_i[test_i:]])
np.save(working_path+"testImages.npy",final_images[rand_i[:test_i]])
np.save(working_path+"testMasks.npy",final_masks[rand_i[:test_i]])


# In[ ]:

np.save(working_path+"myTrainImages.npy",final_images)
np.save(working_path+"myTrainMasks.npy",final_masks)


# In[1]:




# In[ ]:

# my_final_images = np.ndarray([num_images,1,7,41,41],dtype=np.float32)
# for i, img_file in enumerate(tqdm(file_list)):
#     imgs = np.load(img_file)
#     if (7, 41, 41) != imgs.shape:
#         print i
#         print imgs.shape
#     #print final_images[i,0].shape
#     my_final_images[i,0] = imgs
    
# my_fname = working_path+'my_train_image.npy'
# print 'writing %s' % my_fname
# np.save(my_fname,my_final_images)


# In[ ]:

#num_images = 1186
# file_list=glob(working_path+"my_masks_*.npy")

# final_masks = np.ndarray([num_images,1,7,41,41],dtype=np.float32)
# for i, mask_file in enumerate(tqdm(file_list)):
#     masks = np.load(mask_file)
#     if (7, 41, 41) != masks.shape:
#         print i
#         print masks.shape
#     #print final_images[i,0].shape
#     final_masks[i,0] = masks
    
# my_fname = working_path+'my_train_masks.npy'
# print 'writing %s' % my_fname
# np.save(my_fname,final_masks)


# In[ ]:

# rand_i = np.random.choice(range(num_images),size=num_images,replace=False)
# test_i = int(0.2*num_images)
# np.save(working_path+"my_train_images.npy",final_images[rand_i[test_i:]])
# np.save(working_path+"my_train_masks.npy",final_masks[rand_i[test_i:]])
# np.save(working_path+"my_test_images.npy",final_images[rand_i[:test_i]])
# np.save(working_path+"my_test_masks.npy",final_masks[rand_i[:test_i]])

