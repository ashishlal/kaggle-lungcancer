
# coding: utf-8

# In[1]:

import os
#os.environ['THEANO_FLAGS'] = 'floatX=float32,device=gpu0,nvcc.fastmath=True,lib.cnmem=0.85'
#THEANO_FLAGS=device=gpu python -c "import theano; print(theano.sandbox.cuda.device_properties(0))"
import keras
#import theano

import sklearn
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, f1_score, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn import metrics
from skimage.transform import resize

import gzip
import numpy as np
import pandas as pd
import cPickle as pickle
import time
from datetime import datetime
from keras_tqdm import TQDMCallback, TQDMNotebookCallback
try:
    from tqdm import tqdm # long waits are not fun
except:
    print('TQDM does make much nicer wait bars...')
    tqdm = lambda x: x

#import matplotlib
#matplotlib.use("Pdf")
#import matplotlib.pyplot as plt
#%matplotlib inline

#from helper_functions import *

from keras.layers import merge, Convolution2D, MaxPooling2D, Input, UpSampling2D
from keras.layers import merge, Convolution3D, MaxPooling3D, Input, UpSampling3D
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Model
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

K.set_image_dim_ordering('th')  # Theano dimension ordering in this code


# In[2]:

working_path = "/home/watts/lal/Kaggle/lung_cancer/"
model_path = "/home/watts/lal/Kaggle/lung_cancer/models/"


# In[3]:

def check_if_image_exists(fname):
    fname = os.path.join(working_path+'data/stage1/stage1/', fname)
    return os.path.exists(fname)

def check_if_scan_exists(folder):
    folder = os.path.join(working_path+'data/stage1/stage1/', folder)
    return os.path.isdir(folder)

def get_current_date():
    return strftime('%Y%m%d')


def load_images(df):
    for i, row in df['ImageFile'].iterrows():
        img = imread(row)
        yield img


# In[4]:

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


# In[5]:

df = pd.read_csv(working_path+'data/stage1/stage1_labels.csv')

df['scan_folder'] = df['id']

df['exist'] = df['scan_folder'].apply(check_if_scan_exists)

print '%i does not exists' % (len(df) - df['exist'].sum())
print df[~df['exist']]

df = df[df['exist']]
df = df.reset_index(drop=True)
    


# In[6]:

# inputs = Input((1, 16, 128, 128))

# # output W2 = 128 x 128 x 16 x 32
# conv1 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(inputs)

# # output 128 x 128 x 16 x 32
# conv1 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(conv1)

# # output 64 x 64 x 8 x 32
# pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

# # output 64 x 64 x 8 x 64
# conv2 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(pool1)

# # output 64 x 64 x 8 x 64
# conv2 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(conv2)

# # output 32 x 32 x 4 x 64
# pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

# # output 32 x 32 x 4 x 128
# conv3 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(pool2)

# #output 32 x 32 x 4 x 128
# conv3 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(conv3)

# # output 16 x 16 x 2 x 128
# pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

# # output 32 x 32 x 2 x 256
# #conv4 = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same')(pool3)

# #output 32 x 32 x 2 x 256
# #conv4 = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same')(conv4)

# #output 16 x 16 x 1 x 256
# #pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

# #output 16 x 16 x 1 x 512
# #conv5 = Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same')(pool4)

# #output 16 x 16 x 1 x 512
# #conv5 = Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same')(conv5)

# #output of UpSampling2D(size=(2, 2, 2))(conv5): 32 x 32 x 2 x 512
# #outputof up6: 32 x 32 x 2 x 768
# #up6 = merge([UpSampling3D(size=(2, 2, 2))(conv5), conv4], mode='concat', concat_axis=1)

# # output 16 x 16 x 2 x 256
# #conv6 = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same')(up6)
# conv6 = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same')(pool3)

# # output 16 x 16 x 2 x 256
# conv6 = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same')(conv6)

# # output of UpSampling2D(size=(2, 2, 2))(conv6): 32 x 32 x 4x 256
# # outputof up7: 32 x 32 x 4 x 384
# up7 = merge([UpSampling3D(size=(2, 2, 2))(conv6), conv3], mode='concat', concat_axis=1)

# # output 32 x 32 x 4 x 128
# conv7 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(up7)

# # output 32 x 32 x 4 x 128
# conv7 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(conv7)

# # output of UpSampling2D(size=(2, 2, 2))(conv7): 64 x 64 x 8 x 128
# # outputof up8: 64  x 64  x 8 x 192
# up8 = merge([UpSampling3D(size=(2, 2, 2))(conv7), conv2], mode='concat', concat_axis=1)

# # output 64 x 64 x 8 x 64
# conv8 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(up8)

# # output 64 x 64 x 8 x 64
# conv8 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(conv8)

# # output of UpSampling2D(size=(2, 2, 2))(conv8): 128 x 128 x 16 x 64
# # outputof up9: 128 x 128 x 16 x 96
# up9 = merge([UpSampling3D(size=(2, 2, 2))(conv8), conv1], mode='concat', concat_axis=1)

# # output 128 x 128 x 16 x 32
# conv9 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(up9)

# # output 128 x 128 x 16 x 32
# conv9 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(conv9)

# # output 128 x 128 x 16 x 1
# conv10 = Convolution3D(1, 1, 1, 1, activation='sigmoid')(conv9)

# model = Model(input=inputs, output=conv10)


# In[7]:

# from keras.models import load_model

# # returns a compiled model
# model.load_weights(model_path+'unet3d_train.hdf5')
# model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])


# In[8]:

num_slices = 16
img_width = 128
img_height = 128


# In[9]:

model_arch = 'model_unet3d_%d.json' % img_width
model_weights = 'model_unet3d_%d.h5' % img_width


# In[10]:

# load json and create model
from keras.models import model_from_json

json_file = open(model_path+model_arch, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights(model_path+model_weights)
print("Loaded model from disk")


# In[11]:

model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])


# In[12]:

import datetime


# In[13]:

IMG_PX_SIZE = img_width
IMG_PX_SIZE_ORG = 512
HM_SLICES = num_slices
num_zeros = 0
num_test = len(df)
segmented_lung_slices = np.ndarray([num_test,1,num_slices,img_width,img_height],dtype=np.float32)
for i, row in tqdm(df.iterrows(), total=len(df)):
    #if i != 0:
    #    continue
    fname = os.path.join('data/stage1/stage1/', row['scan_folder'])
    try:
        j = 0
        scan_folder = row['scan_folder']

        X_segmented_lung_fname = working_path+'cache/segmentation_1_1_1/X_segmented_lungs_%s_%s_%s_%s.npy' % (scan_folder, HM_SLICES, IMG_PX_SIZE, IMG_PX_SIZE)
        X_nodule_fname = working_path+'cache/X_nodule_%s_%s_%s_%s.npy' % (scan_folder, HM_SLICES, IMG_PX_SIZE, IMG_PX_SIZE)
        #print '0..'
        segmented_lung = np.load(X_segmented_lung_fname)
        #print segmented_lung.shape
        #segmented_lung = resize(segmented_lung,[16, 128,128])
        segmented_lung = segmented_lung[0]
        #print segmented_lung.shape
        
        nz = np.count_nonzero(segmented_lung)
        if nz == 0:
            print 'slice is 0...'
            num_zeros += 1
        #print '1..'
        segmented_lung_slices[i,0] = segmented_lung
        my_nodules_mask = model.predict([segmented_lung_slices[i:i+1]], verbose=0)[0]
        #print '2..'
        np.save(X_nodule_fname, my_nodules_mask)
    except:
        print '%s has failed' % i
        #sys.exit(1)
print 'Done'
now = datetime.datetime.now()
print now


# In[14]:

df = pd.read_csv(working_path+'data/stage1/stage1_sample_submission.csv')

df['scan_folder'] = df['id']

df['exist'] = df['scan_folder'].apply(check_if_scan_exists)

print '%i does not exists' % (len(df) - df['exist'].sum())
print df[~df['exist']]

df = df[df['exist']]
df = df.reset_index(drop=True)
    


# In[15]:

IMG_PX_SIZE = img_width
IMG_PX_SIZE_ORG = 512
HM_SLICES = num_slices
num_zeros = 0
num_test = len(df)
#print num_test
test_lung_slices = np.ndarray([num_test,1,num_slices,img_width,img_height],dtype=np.float32)
for i, row in tqdm(df.iterrows(), total=len(df)):
#     if i != 0:
#        continue
    fname = os.path.join('data/stage1/stage1/', row['scan_folder'])
    #print fname
    try:
        j = 0
        scan_folder = row['scan_folder']
        #print scan_folder
        X_test_segmented_lung_fname = working_path+'cache/segmentation_1_1_1/X_test_segmented_lungs_%s_%s_%s_%s.npy' % (scan_folder, HM_SLICES, IMG_PX_SIZE, IMG_PX_SIZE)
        #print '0a..'
        #print(X_test_segmented_lung_fname)
        X_test_nodule_fname = working_path+'cache/X_test_nodule_%s_%s_%s_%s.npy' % (scan_folder, HM_SLICES, IMG_PX_SIZE, IMG_PX_SIZE)
        #print '0..'
        segmented_lung = np.load(X_test_segmented_lung_fname)
        #print segmented_lung.shape
        #segmented_lung = resize(segmented_lung,[16, 128,128])
        segmented_lung = segmented_lung[0]
        #print segmented_lung.shape
        
        nz = np.count_nonzero(segmented_lung)
        if nz == 0:
            print 'slice is 0...'
            num_zeros += 1
            continue
        #print '1..'
        test_lung_slices[i,0] = segmented_lung
        my_nodules_mask = model.predict([test_lung_slices[i:i+1]], verbose=0)[0]
        #print '2..'
        np.save(X_test_nodule_fname, my_nodules_mask)
    except:
        print '%s has failed' % i
        #sys.exit(1)
print 'Done'
now = datetime.datetime.now()
print now


# In[ ]:



