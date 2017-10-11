
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

num_slices = 16
img_width = 128
img_height = 128


# In[7]:

model_arch = 'model_unet_%d.json' % img_width
model_weights = 'model_unet_%d.h5' % img_width


# In[8]:

# load json and create model
from keras.models import model_from_json

json_file = open(model_path+model_arch, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights(model_path+model_weights)
print("Loaded model from disk")


# In[9]:

model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])


# In[10]:

import datetime


# In[11]:

IMG_PX_SIZE = img_width
IMG_PX_SIZE_ORG = 512
HM_SLICES = num_slices
num_zeros = 0
num_test = 1397 * num_slices
segmented_lung_slices = np.ndarray([2,1,img_width,img_height],dtype=np.float32)
my_nodules_mask = np.ndarray([num_slices,1,img_width,img_height],dtype=np.float32)

for i, row in tqdm(df.iterrows(), total=len(df)):
    #if i != 0:
    #    continue
    try:
        scan_folder = row['scan_folder']
        #print scan_folder
        for j in range(num_slices):
            X_segmented_lung_fname = working_path+'cache/X_segmented_lung_%s_%d_%d_%d.npy' % (scan_folder, j, IMG_PX_SIZE, IMG_PX_SIZE)
            #print '0..'
            segmented_lung = np.load(X_segmented_lung_fname)
            #print segmented_lung.shape
            #segmented_lung = resize(segmented_lung,[16, 128,128])
            #print '1..'
            segmented_lung_slices[0,0] = segmented_lung
            my_nm = model.predict([segmented_lung_slices[0:1]], verbose=0)[0]
            #print my_nm.shape
            my_nodules_mask[j] = my_nm
            #print '2..'
        X_nodule_fname = working_path+'cache/X_nodule_%s_%d_%d.npy' % (scan_folder, IMG_PX_SIZE, IMG_PX_SIZE)
        np.save(X_nodule_fname, my_nodules_mask)
    except:
        print '%s has failed' % i
        #sys.exit(1)
print 'Done'
now = datetime.datetime.now()
print now


# In[12]:

df = pd.read_csv(working_path+'data/stage1/stage1_sample_submission.csv')

df['scan_folder'] = df['id']

df['exist'] = df['scan_folder'].apply(check_if_scan_exists)

print '%i does not exists' % (len(df) - df['exist'].sum())
print df[~df['exist']]

df = df[df['exist']]
df = df.reset_index(drop=True)
    


# In[13]:

IMG_PX_SIZE = img_width
IMG_PX_SIZE_ORG = 512
HM_SLICES = num_slices
num_zeros = 0
num_test = 198 * num_slices
#print num_test
test_lung_slices = np.ndarray([1,1,img_width,img_height],dtype=np.float32)
my_nodules_mask = np.ndarray([num_slices,1,img_width,img_height],dtype=np.float32)

for i, row in tqdm(df.iterrows(), total=len(df)):
    #if i != 0:
    #    continue
    try:
        scan_folder = row['scan_folder']
        for j in range(num_slices):
            X_segmented_lung_fname = working_path+'cache/X_test_segmented_lung_%s_%d_%d_%d.npy' % (scan_folder, j, IMG_PX_SIZE, IMG_PX_SIZE)
            #print '0..'
            segmented_lung = np.load(X_segmented_lung_fname)
            #print segmented_lung.shape
            #segmented_lung = resize(segmented_lung,[16, 128,128])
            #print '1..'
            segmented_lung_slices[0,0] = segmented_lung
            my_nm = model.predict([segmented_lung_slices[0:1]], verbose=0)[0]
            #print my_nm.shape
            my_nodules_mask[j] = my_nm
            #print '2..'
        X_nodule_fname = working_path+'cache/X_test_nodule_%s_%d_%d.npy' % (scan_folder, IMG_PX_SIZE, IMG_PX_SIZE)
        np.save(X_nodule_fname, my_nodules_mask)
    except:
        print '%s has failed' % i
        #sys.exit(1)
print 'Done'
now = datetime.datetime.now()
print now


# In[ ]:



