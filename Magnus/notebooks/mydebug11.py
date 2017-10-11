
# coding: utf-8

# In[1]:

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
import keras
from keras import backend as K
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
K.set_image_dim_ordering('th')


# In[2]:

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_np(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# In[3]:

working_path = "/home/watts/lal/Kaggle/lung_cancer/"
model_path = "/home/watts/lal/Kaggle/lung_cancer/models/"


# In[4]:

from keras.models import model_from_json

model_path = "/home/watts/lal/Kaggle/lung_cancer/models/"

json_file = open(model_path+'model_vnet_128.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights(model_path+"model_vnet_128.h5")
print("Loaded model from disk")


# In[5]:

from keras.optimizers import Adam
model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])


# In[6]:

X_segmented_lung_fname = working_path+'cache/X_segmented_lungs_0015ceb851d7251b8f399e39779d1e7d_16_128_128.npy'
segmented_lung = np.load(X_segmented_lung_fname)


# In[7]:

imgs1 = segmented_lung
fig,ax = plt.subplots(4,4,figsize=[8,8])
#for i in range(imgs1.shape[1]):
    #print "image %d" % i
    
ax[0,0].imshow(imgs1[0,0],cmap='gray')
ax[0,1].imshow(imgs1[0,1],cmap='gray')
ax[0,2].imshow(imgs1[0,2],cmap='gray')
ax[0,3].imshow(imgs1[0,3],cmap='gray')

ax[1,0].imshow(imgs1[0,4],cmap='gray')
ax[1,1].imshow(imgs1[0,5],cmap='gray')
ax[1,2].imshow(imgs1[0,6],cmap='gray')
ax[1,3].imshow(imgs1[0,7],cmap='gray')

ax[2,0].imshow(imgs1[0,8],cmap='gray')
ax[2,1].imshow(imgs1[0,9],cmap='gray')
ax[2,2].imshow(imgs1[0,10],cmap='gray')
ax[2,3].imshow(imgs1[0,11],cmap='gray')

ax[3,0].imshow(imgs1[0,12],cmap='gray')
ax[3,1].imshow(imgs1[0,13],cmap='gray')
ax[3,2].imshow(imgs1[0,14],cmap='gray')
ax[3,3].imshow(imgs1[0,15],cmap='gray')
plt.show()
#raw_input("hit enter to cont : ")


# In[8]:

nodules_mask = np.ndarray([1,1,16,128,128],dtype=np.float32)
#X_segmented_lung_fname = working_path+'cache/X_segmented_lungs_0015ceb851d7251b8f399e39779d1e7d_16_128_128.npy'
#segmented_lung = np.load(X_segmented_lung_fname)
print segmented_lung.shape
#segmented_lung = resize(segmented_lung,[16, 128,128])
segmented_lung = segmented_lung[0]
print segmented_lung.shape
i = 0
nz = np.count_nonzero(segmented_lung)
if nz == 0:
    print 'slice is 0...'
print '1..'
nodules_mask[0,0] = segmented_lung
my_nodules_mask = model.predict([nodules_mask[0:1]], verbose=0)[0]
print '2..'
nz = np.count_nonzero(my_nodules_mask)
if nz == 0:
    print 'mask is 0...'
#np.save(X_nodule_fname, my_nodules_mask)


# In[9]:

nz


# In[10]:

print my_nodules_mask.shape


# In[11]:

nz = np.count_nonzero(my_nodules_mask)
print nz
for j in range(16):
    img = my_nodules_mask[0,j]
    nz = np.count_nonzero(img)
    print nz
imgs1 = my_nodules_mask
fig,ax = plt.subplots(4,4,figsize=[8,8])
#for i in range(imgs1.shape[1]):
    #print "image %d" % i
    
ax[0,0].imshow(imgs1[0,0],cmap='gray')
ax[0,1].imshow(imgs1[0,1],cmap='gray')
ax[0,2].imshow(imgs1[0,2],cmap='gray')
ax[0,3].imshow(imgs1[0,3],cmap='gray')

ax[1,0].imshow(imgs1[0,4],cmap='gray')
ax[1,1].imshow(imgs1[0,5],cmap='gray')
ax[1,2].imshow(imgs1[0,6],cmap='gray')
ax[1,3].imshow(imgs1[0,7],cmap='gray')

ax[2,0].imshow(imgs1[0,8],cmap='gray')
ax[2,1].imshow(imgs1[0,9],cmap='gray')
ax[2,2].imshow(imgs1[0,10],cmap='gray')
ax[2,3].imshow(imgs1[0,11],cmap='gray')

ax[3,0].imshow(imgs1[0,12],cmap='gray')
ax[3,1].imshow(imgs1[0,13],cmap='gray')
ax[3,2].imshow(imgs1[0,14],cmap='gray')
ax[3,3].imshow(imgs1[0,15],cmap='gray')
plt.show()
#raw_input("hit enter to cont : ")


# In[ ]:




# In[33]:

X_nodule_mask_fname = working_path+'cache/X_nodule_0015ceb851d7251b8f399e39779d1e7d_16_128_128.npy'
my_nodules_mask1 = np.load(X_nodule_mask_fname)


# In[34]:

print my_nodules_mask1.shape


# In[35]:

imgs1 = my_nodules_mask1
fig,ax = plt.subplots(4,4,figsize=[8,8])
#for i in range(imgs1.shape[1]):
    #print "image %d" % i
    
ax[0,0].imshow(imgs1[0,0],cmap='gray')
ax[0,1].imshow(imgs1[0,1],cmap='gray')
ax[0,2].imshow(imgs1[0,2],cmap='gray')
ax[0,3].imshow(imgs1[0,3],cmap='gray')

ax[1,0].imshow(imgs1[0,4],cmap='gray')
ax[1,1].imshow(imgs1[0,5],cmap='gray')
ax[1,2].imshow(imgs1[0,6],cmap='gray')
ax[1,3].imshow(imgs1[0,7],cmap='gray')

ax[2,0].imshow(imgs1[0,8],cmap='gray')
ax[2,1].imshow(imgs1[0,9],cmap='gray')
ax[2,2].imshow(imgs1[0,10],cmap='gray')
ax[2,3].imshow(imgs1[0,11],cmap='gray')

ax[3,0].imshow(imgs1[0,12],cmap='gray')
ax[3,1].imshow(imgs1[0,13],cmap='gray')
ax[3,2].imshow(imgs1[0,14],cmap='gray')
ax[3,3].imshow(imgs1[0,15],cmap='gray')
plt.show()
raw_input("hit enter to cont : ")


# In[16]:

working_path1 = "/home/watts/lal/Kaggle/lung_cancer/cache/luna16/512_512_16/"
model_path = "/home/watts/lal/Kaggle/lung_cancer/models/"


# In[17]:

imgs_train = np.load(working_path1+"my_train_images_16_128_128.npy").astype(np.float32)
imgs_mask_train = np.load(working_path1+"my_train_masks_16_128_128.npy").astype(np.float32)

imgs_test = np.load(working_path1+"my_test_images_16_128_128.npy").astype(np.float32)
imgs_mask_test_true = np.load(working_path1+"my_test_masks_16_128_128.npy").astype(np.float32)
    
mean = np.mean(imgs_train)  # mean for data centering
std = np.std(imgs_train)  # std for data normalization

imgs_train -= mean  # images should already be standardized, but just in case
imgs_train /= std

train_x = imgs_train
train_y = imgs_mask_train

test_x = imgs_test
test_y = imgs_mask_test_true


# In[18]:

print imgs_test.shape


# In[19]:

imgs1 = imgs_test[1]
fig,ax = plt.subplots(4,4,figsize=[8,8])
#for i in range(imgs1.shape[1]):
    #print "image %d" % i
    
ax[0,0].imshow(imgs1[0,0],cmap='gray')
ax[0,1].imshow(imgs1[0,1],cmap='gray')
ax[0,2].imshow(imgs1[0,2],cmap='gray')
ax[0,3].imshow(imgs1[0,3],cmap='gray')

ax[1,0].imshow(imgs1[0,4],cmap='gray')
ax[1,1].imshow(imgs1[0,5],cmap='gray')
ax[1,2].imshow(imgs1[0,6],cmap='gray')
ax[1,3].imshow(imgs1[0,7],cmap='gray')

ax[2,0].imshow(imgs1[0,8],cmap='gray')
ax[2,1].imshow(imgs1[0,9],cmap='gray')
ax[2,2].imshow(imgs1[0,10],cmap='gray')
ax[2,3].imshow(imgs1[0,11],cmap='gray')

ax[3,0].imshow(imgs1[0,12],cmap='gray')
ax[3,1].imshow(imgs1[0,13],cmap='gray')
ax[3,2].imshow(imgs1[0,14],cmap='gray')
ax[3,3].imshow(imgs1[0,15],cmap='gray')
plt.show()
#raw_input("hit enter to cont : ")


# In[14]:

print imgs_mask_test_true.shape


# In[20]:

imgs1 = imgs_mask_test_true[1]
fig,ax = plt.subplots(4,4,figsize=[8,8])
#for i in range(imgs1.shape[1]):
    #print "image %d" % i
    
ax[0,0].imshow(imgs1[0,0],cmap='gray')
ax[0,1].imshow(imgs1[0,1],cmap='gray')
ax[0,2].imshow(imgs1[0,2],cmap='gray')
ax[0,3].imshow(imgs1[0,3],cmap='gray')

ax[1,0].imshow(imgs1[0,4],cmap='gray')
ax[1,1].imshow(imgs1[0,5],cmap='gray')
ax[1,2].imshow(imgs1[0,6],cmap='gray')
ax[1,3].imshow(imgs1[0,7],cmap='gray')

ax[2,0].imshow(imgs1[0,8],cmap='gray')
ax[2,1].imshow(imgs1[0,9],cmap='gray')
ax[2,2].imshow(imgs1[0,10],cmap='gray')
ax[2,3].imshow(imgs1[0,11],cmap='gray')

ax[3,0].imshow(imgs1[0,12],cmap='gray')
ax[3,1].imshow(imgs1[0,13],cmap='gray')
ax[3,2].imshow(imgs1[0,14],cmap='gray')
ax[3,3].imshow(imgs1[0,15],cmap='gray')
plt.show()


# In[21]:

imgs_mask_test = np.ndarray([1,1,16,128,128],dtype=np.float32)
#for i in range(num_test):
i = 0
imgs_mask_test[i] = model.predict([imgs_test[i:i+1]], verbose=0)[0]


# In[22]:

imgs1 = imgs_mask_test[0]
fig,ax = plt.subplots(4,4,figsize=[8,8])
#for i in range(imgs1.shape[1]):
    #print "image %d" % i
    
ax[0,0].imshow(imgs1[0,0],cmap='gray')
ax[0,1].imshow(imgs1[0,1],cmap='gray')
ax[0,2].imshow(imgs1[0,2],cmap='gray')
ax[0,3].imshow(imgs1[0,3],cmap='gray')

ax[1,0].imshow(imgs1[0,4],cmap='gray')
ax[1,1].imshow(imgs1[0,5],cmap='gray')
ax[1,2].imshow(imgs1[0,6],cmap='gray')
ax[1,3].imshow(imgs1[0,7],cmap='gray')

ax[2,0].imshow(imgs1[0,8],cmap='gray')
ax[2,1].imshow(imgs1[0,9],cmap='gray')
ax[2,2].imshow(imgs1[0,10],cmap='gray')
ax[2,3].imshow(imgs1[0,11],cmap='gray')

ax[3,0].imshow(imgs1[0,12],cmap='gray')
ax[3,1].imshow(imgs1[0,13],cmap='gray')
ax[3,2].imshow(imgs1[0,14],cmap='gray')
ax[3,3].imshow(imgs1[0,15],cmap='gray')
plt.show()


# In[24]:

num_slices = 16
img_width = 128
img_height = 128
my_train_img_fname = 'my_t_image_%d_%d_%d.npy' % (num_slices, img_width, img_height)
my_train_mask_fname = 'my_t_mask_%d_%d_%d.npy' % (num_slices, img_width, img_height)


my_imgs_train = np.load(working_path1+my_train_img_fname).astype(np.float32)
my_imgs_mask_train = np.load(working_path1+my_train_mask_fname).astype(np.float32)


mean = np.mean(my_imgs_train)  # mean for data centering
std = np.std(my_imgs_train)  # std for data normalization

my_imgs_train -= mean  # images should already be standardized, but just in case
my_imgs_train /= std

train_x = my_imgs_train
train_y = my_imgs_mask_train


# In[29]:

imgs1 = train_x[1]
fig,ax = plt.subplots(4,4,figsize=[8,8])
#for i in range(imgs1.shape[1]):
    #print "image %d" % i
    
ax[0,0].imshow(imgs1[0,0],cmap='gray')
ax[0,1].imshow(imgs1[0,1],cmap='gray')
ax[0,2].imshow(imgs1[0,2],cmap='gray')
ax[0,3].imshow(imgs1[0,3],cmap='gray')

ax[1,0].imshow(imgs1[0,4],cmap='gray')
ax[1,1].imshow(imgs1[0,5],cmap='gray')
ax[1,2].imshow(imgs1[0,6],cmap='gray')
ax[1,3].imshow(imgs1[0,7],cmap='gray')

ax[2,0].imshow(imgs1[0,8],cmap='gray')
ax[2,1].imshow(imgs1[0,9],cmap='gray')
ax[2,2].imshow(imgs1[0,10],cmap='gray')
ax[2,3].imshow(imgs1[0,11],cmap='gray')

ax[3,0].imshow(imgs1[0,12],cmap='gray')
ax[3,1].imshow(imgs1[0,13],cmap='gray')
ax[3,2].imshow(imgs1[0,14],cmap='gray')
ax[3,3].imshow(imgs1[0,15],cmap='gray')
plt.show()
#raw_input("hit enter to cont : ")


# In[28]:

imgs1 = train_y[1]
fig,ax = plt.subplots(4,4,figsize=[8,8])
#for i in range(imgs1.shape[1]):
    #print "image %d" % i
    
ax[0,0].imshow(imgs1[0,0],cmap='gray')
ax[0,1].imshow(imgs1[0,1],cmap='gray')
ax[0,2].imshow(imgs1[0,2],cmap='gray')
ax[0,3].imshow(imgs1[0,3],cmap='gray')

ax[1,0].imshow(imgs1[0,4],cmap='gray')
ax[1,1].imshow(imgs1[0,5],cmap='gray')
ax[1,2].imshow(imgs1[0,6],cmap='gray')
ax[1,3].imshow(imgs1[0,7],cmap='gray')

ax[2,0].imshow(imgs1[0,8],cmap='gray')
ax[2,1].imshow(imgs1[0,9],cmap='gray')
ax[2,2].imshow(imgs1[0,10],cmap='gray')
ax[2,3].imshow(imgs1[0,11],cmap='gray')

ax[3,0].imshow(imgs1[0,12],cmap='gray')
ax[3,1].imshow(imgs1[0,13],cmap='gray')
ax[3,2].imshow(imgs1[0,14],cmap='gray')
ax[3,3].imshow(imgs1[0,15],cmap='gray')
plt.show()
#raw_input("hit enter to cont : ")


# In[44]:

for i in range(16):
    nz = np.count_nonzero(imgs1[0,i])
    if nz == 0:
        print 'slice is 0...'


# In[45]:

print imgs_mask_train.shape


# In[46]:

imgs1 = imgs_mask_train[0]
fig,ax = plt.subplots(4,4,figsize=[8,8])
#for i in range(imgs1.shape[1]):
    #print "image %d" % i
    
ax[0,0].imshow(imgs1[0,0],cmap='gray')
ax[0,1].imshow(imgs1[0,1],cmap='gray')
ax[0,2].imshow(imgs1[0,2],cmap='gray')
ax[0,3].imshow(imgs1[0,3],cmap='gray')

ax[1,0].imshow(imgs1[0,4],cmap='gray')
ax[1,1].imshow(imgs1[0,5],cmap='gray')
ax[1,2].imshow(imgs1[0,6],cmap='gray')
ax[1,3].imshow(imgs1[0,7],cmap='gray')

ax[2,0].imshow(imgs1[0,8],cmap='gray')
ax[2,1].imshow(imgs1[0,9],cmap='gray')
ax[2,2].imshow(imgs1[0,10],cmap='gray')
ax[2,3].imshow(imgs1[0,11],cmap='gray')

ax[3,0].imshow(imgs1[0,12],cmap='gray')
ax[3,1].imshow(imgs1[0,13],cmap='gray')
ax[3,2].imshow(imgs1[0,14],cmap='gray')
ax[3,3].imshow(imgs1[0,15],cmap='gray')
plt.show()


# In[47]:

print imgs_mask_train.shape


# In[48]:

print 949 * 16


# In[49]:

num_zero = 0
for i in range(imgs_mask_train.shape[0]):
    for j in range(imgs_mask_train.shape[2]):
        img = imgs_mask_train[i]
        nz = np.count_nonzero(img[0,j])
        if nz == 0:
        #    print 'slice is 0...'
            num_zero += 1
print num_zero


# In[50]:

num_zero = 0
for i in range(imgs_mask_train.shape[0]):
    img = imgs_mask_train[i]
    nz = np.count_nonzero(img)
    if nz == 0:
    #    print 'slice is 0...'
        num_zero += 1
print num_zero


# In[53]:

print imgs_mask_test_true.shape


# In[54]:

print 237 * 16


# In[55]:

num_zero = 0
for i in range(imgs_mask_test_true.shape[0]):
    for j in range(imgs_mask_test_true.shape[2]):
        img = imgs_mask_test_true[i]
        nz = np.count_nonzero(img[0,j])
        if nz == 0:
        #    print 'slice is 0...'
            num_zero += 1
print num_zero


# In[56]:

print imgs_train.shape


# In[57]:

print 949 * 16


# In[58]:

num_zero = 0
for i in range(imgs_train.shape[0]):
    for j in range(imgs_train.shape[2]):
        img = imgs_train[i]
        nz = np.count_nonzero(img[0,j])
        if nz == 0:
        #    print 'slice is 0...'
            num_zero += 1
print num_zero


# In[59]:

my_imgs_train = np.load(working_path1+"my_image_16_128_128.npy").astype(np.float32)
my_imgs_mask_train = np.load(working_path1+"my_mask_16_128_128.npy").astype(np.float32)


mean = np.mean(my_imgs_train)  # mean for data centering
std = np.std(my_imgs_train)  # std for data normalization

my_imgs_train -= mean  # images should already be standardized, but just in case
my_imgs_train /= std


# In[60]:

print my_imgs_train.shape


# In[61]:

print 1186 * 16


# In[62]:

num_zero = 0
for i in range(my_imgs_train.shape[0]):
    for j in range(my_imgs_train.shape[2]):
        img = my_imgs_train[i]
        nz = np.count_nonzero(img[0,j])
        if nz == 0:
        #    print 'slice is 0...'
            num_zero += 1
print num_zero


# In[63]:

print my_imgs_mask_train.shape


# In[64]:

print 1186  * 16


# In[65]:

num_zero = 0
for i in range(my_imgs_mask_train.shape[0]):
    #for j in range(my_imgs_mask_train.shape[2]):
    img = my_imgs_mask_train[i]
    nz = np.count_nonzero(img)
    if nz == 0:
    #    print 'slice is 0...'
        num_zero += 1
print num_zero


# In[66]:

num_zero = 0
for i in range(my_imgs_mask_train.shape[0]):
    for j in range(my_imgs_mask_train.shape[2]):
        img = my_imgs_mask_train[i]
        nz = np.count_nonzero(img[0,j])
        if nz == 0:
        #    print 'slice is 0...'
            num_zero += 1
print num_zero


# In[ ]:



