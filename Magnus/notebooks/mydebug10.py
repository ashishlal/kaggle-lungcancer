
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

import gzip
import numpy as np
import pandas as pd
import cPickle as pickle
import time
from datetime import datetime


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


# In[11]:

working_path = "/home/watts/lal/Kaggle/lung_cancer/cache/luna16/512_512_16/"
model_path = "/home/watts/lal/Kaggle/lung_cancer/models/"
#working_path = "/home/watts/lal/Kaggle/lung_cancer/cache/luna16/"


# In[3]:

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


# In[4]:

def get_weights(layer):
     try:
        w = layer.get_weights()
        print w
        weights = layer.get_weights()[0]
        biases = layer.get_weights()[1]
        counts = np.prod(weights.shape) + np.prod(biases.shape)
        print counts
     except:
        print('Layer has no weights')


# In[5]:

imgs_train = np.load(working_path+"my_train_images_16_128_128.npy").astype(np.float32)
imgs_mask_train = np.load(working_path+"my_train_masks_16_128_128.npy").astype(np.float32)

imgs_test = np.load(working_path+"my_test_images_16_128_128.npy").astype(np.float32)
imgs_mask_test_true = np.load(working_path+"my_test_masks_16_128_128.npy").astype(np.float32)
    
mean = np.mean(imgs_train)  # mean for data centering
std = np.std(imgs_train)  # std for data normalization

imgs_train -= mean  # images should already be standardized, but just in case
imgs_train /= std

train_x = imgs_train
train_y = imgs_mask_train

test_x = imgs_test
test_y = imgs_mask_test_true


# In[6]:

train_x.shape


# In[1]:

#train_y.shape


# In[30]:

#train_x = np.concatenate( [imgs_train[i] for i in range(imgs_train.shape[0])] )
#train_y = np.concatenate([imgs_mask_train[i] for i in range(imgs_mask_train.shape[0])])
#est_x = np.concatenate( [imgs_test[i] for i in range(imgs_test.shape[0])] )
#test_y = np.concatenate([imgs_mask_test_true[i] for i in range(imgs_mask_test_true.shape[0])])


# In[7]:

print train_x.shape


# In[8]:

print train_y.shape


# In[33]:

#train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1], train_x.shape[2],train_x.shape[3])

#test_x = test_x.reshape(test_x.shape[0], 1, test_x.shape[1], test_x.shape[2],test_x.shape[3])


# In[34]:

#train_y = np.array([train_y[i] for i in range(len(train_y))])

#test_y = np.array([test_y[i] for i in range(len(test_y))])


# In[35]:

#train_y = train_y.reshape(train_y.shape[0], 1, train_y.shape[1], train_y.shape[2],train_y.shape[3])
#test_y = test_y.reshape(test_y.shape[0], 1, test_y.shape[1], test_y.shape[2],test_y.shape[3])


# In[36]:

#train_y.shape


# In[37]:

#train_x.shape


# In[ ]:

# Convolution
#Accepts a volume of size W1×H1×D1
#Requires four hyperparameters:
#Number of filters K,
#their spatial extent F,
#the stride S,
#the amount of zero padding P.
#Produces a volume of size W2×H2×D2 where:
#W2=(W1−F+2P)/S+1
#H2=(H1−F+2P)/S+1 (i.e. width and height are computed equally by symmetry)
#D2=K
# With parameter sharing, it introduces F*F*D1 weights per filter, for a total of (F⋅F⋅D1)⋅K weights and K biases.
#In the output volume, the d-th depth slice (of size W2×H2) is the result of performing a valid convolution of the d-th filter over the input volume with a stride of SS, and then offset by dd-th bias.

# Pooling
# Accepts a volume of size W1×H1×D1
# Requires two hyperparameters:
# their spatial extent F,
# the stride S,
# Produces a volume of size W2×H2×D2 where:
# W2=(W1−F)/S+1
# H2=(H1−F)/S+1
# D2=D1


# In[10]:

inputs = Input((1, 16, 512, 512))

# output W2 = 512 x 512 x 16 x 32
conv1 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(inputs)

# output 512 x 512 x 16 x 32
conv1 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(conv1)

# output 256 x 256 x 8 x 32
pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

# output 256 x 256 x 8 x 64
conv2 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(pool1)

# output 256 x 256 x 8 x 64
conv2 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(conv2)

# output 128 x 128 x 4 x 64
pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

# output 128 x 128 x 4 x 128
conv3 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(pool2)

#output 128 x 128 x 4 x 128
conv3 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(conv3)

# output 64 x 64 x 2 x 128
pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

# output 64 x 64 x 2 x 256
conv4 = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same')(pool3)

#output 64 x 64 x 2 x 256
conv4 = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same')(conv4)

#output 32 x 32 x 1 x 256
pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

#output 32 x 32 x 1 x 512
conv5 = Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same')(pool4)

#output 32 x 32 x 1 x 512
conv5 = Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same')(conv5)

#output of UpSampling2D(size=(2, 2, 2))(conv5): 64 x 64 x 2 x 512
#outputof up6: 64 x 64 x 2 x 768
up6 = merge([UpSampling3D(size=(2, 2, 2))(conv5), conv4], mode='concat', concat_axis=1)

# output 64 x 64 x 2 x 256
conv6 = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same')(up6)
#conv6 = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same')(pool4)

# output 64 x 64 x 2 x 256
conv6 = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same')(conv6)

# output of UpSampling2D(size=(2, 2, 2))(conv6): 128 x 128 x 4 x 256
# outputof up7: 128 x 128 x 4 x 384
up7 = merge([UpSampling3D(size=(2, 2, 2))(conv6), conv3], mode='concat', concat_axis=1)

# output 128 x 128 x 4 x 128
conv7 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(up7)

# output 128 x 128 x 4 x 128
conv7 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(conv7)

# output of UpSampling2D(size=(2, 2, 2))(conv7): 256 x 256 x 8 x 128
# outputof up8: 256 x 256 x 8 x 192
up8 = merge([UpSampling3D(size=(2, 2, 2))(conv7), conv2], mode='concat', concat_axis=1)

# output 256 x 256 x 8 x 64
conv8 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(up8)

# output 256 x 256 x 8 x 64
conv8 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(conv8)

# output of UpSampling2D(size=(2, 2, 2))(conv8): 512 x 512 x 16 x 64
# outputof up9: 512 x 512 x 16 x 96
up9 = merge([UpSampling3D(size=(2, 2, 2))(conv8), conv1], mode='concat', concat_axis=1)

# output 512 x 512 x 16 x 32
conv9 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(up9)

# output 512 x 512 x 16 x 32
conv9 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(conv9)

# output 512 x 512 x 16 x 1
conv10 = Convolution3D(1, 1, 1, 1, activation='sigmoid')(conv9)

model = Model(input=inputs, output=conv10)

model.summary()


# In[9]:

inputs = Input((1, 16, 256, 256))

# output W2 = 256 x 256 x 16 x 32
conv1 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(inputs)

# output 256 x 256 x 16 x 32
conv1 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(conv1)

# output 128 x 128 x 8 x 32
pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

# output 128 x 128 x 8 x 64
conv2 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(pool1)

# output 128 x 128 x 8 x 64
conv2 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(conv2)

# output 64 x 64 x 4 x 64
pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

# output 64 x 64 x 4 x 128
conv3 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(pool2)

#output 64 x 64 x 4 x 128
conv3 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(conv3)

# output 32 x 32 x 2 x 128
pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

# output 32 x 32 x 2 x 256
#conv4 = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same')(pool3)

#output 32 x 32 x 2 x 256
#conv4 = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same')(conv4)

#output 16 x 16 x 1 x 256
#pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

#output 16 x 16 x 1 x 512
#conv5 = Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same')(pool4)

#output 16 x 16 x 1 x 512
#conv5 = Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same')(conv5)

#output of UpSampling2D(size=(2, 2, 2))(conv5): 32 x 32 x 2 x 512
#outputof up6: 32 x 32 x 2 x 768
#up6 = merge([UpSampling3D(size=(2, 2, 2))(conv5), conv4], mode='concat', concat_axis=1)

# output 32 x 32 x 2 x 256
#conv6 = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same')(up6)
conv6 = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same')(pool3)

# output 32 x 32 x 2 x 256
conv6 = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same')(conv6)

# output of UpSampling2D(size=(2, 2, 2))(conv6): 64 x 64 x 4 x 256
# outputof up7: 64 x 64 x 4 x 384
up7 = merge([UpSampling3D(size=(2, 2, 2))(conv6), conv3], mode='concat', concat_axis=1)

# output 64 x 64 x 4 x 128
conv7 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(up7)

# output 64 x 64 x 4 x 128
conv7 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(conv7)

# output of UpSampling2D(size=(2, 2, 2))(conv7): 128 x 128 x 8 x 128
# outputof up8: 128  x 128  x 8 x 192
up8 = merge([UpSampling3D(size=(2, 2, 2))(conv7), conv2], mode='concat', concat_axis=1)

# output 128 x 128 x 8 x 64
conv8 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(up8)

# output 128 x 128 x 8 x 64
conv8 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(conv8)

# output of UpSampling2D(size=(2, 2, 2))(conv8): 256 x 256 x 16 x 64
# outputof up9: 256 x 256 x 16 x 96
up9 = merge([UpSampling3D(size=(2, 2, 2))(conv8), conv1], mode='concat', concat_axis=1)

# output 256 x 256 x 16 x 32
conv9 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(up9)

# output 256 x 256 x 16 x 32
conv9 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(conv9)

# output 256 x 256 x 16 x 1
conv10 = Convolution3D(1, 1, 1, 1, activation='sigmoid')(conv9)

model = Model(input=inputs, output=conv10)

model.summary()


# In[7]:

inputs = Input((1, 16, 128, 128))

# output W2 = 128 x 128 x 16 x 32
conv1 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(inputs)

# output 128 x 128 x 16 x 32
conv1 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(conv1)

# output 64 x 64 x 8 x 32
pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

# output 64 x 64 x 8 x 64
conv2 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(pool1)

# output 64 x 64 x 8 x 64
conv2 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(conv2)

# output 32 x 32 x 4 x 64
pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

# output 32 x 32 x 4 x 128
conv3 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(pool2)

#output 32 x 32 x 4 x 128
conv3 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(conv3)

# output 16 x 16 x 2 x 128
pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

# output 32 x 32 x 2 x 256
#conv4 = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same')(pool3)

#output 32 x 32 x 2 x 256
#conv4 = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same')(conv4)

#output 16 x 16 x 1 x 256
#pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

#output 16 x 16 x 1 x 512
#conv5 = Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same')(pool4)

#output 16 x 16 x 1 x 512
#conv5 = Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same')(conv5)

#output of UpSampling2D(size=(2, 2, 2))(conv5): 32 x 32 x 2 x 512
#outputof up6: 32 x 32 x 2 x 768
#up6 = merge([UpSampling3D(size=(2, 2, 2))(conv5), conv4], mode='concat', concat_axis=1)

# output 16 x 16 x 2 x 256
#conv6 = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same')(up6)
conv6 = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same')(pool3)

# output 16 x 16 x 2 x 256
conv6 = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same')(conv6)

# output of UpSampling2D(size=(2, 2, 2))(conv6): 32 x 32 x 4x 256
# outputof up7: 32 x 32 x 4 x 384
up7 = merge([UpSampling3D(size=(2, 2, 2))(conv6), conv3], mode='concat', concat_axis=1)

# output 32 x 32 x 4 x 128
conv7 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(up7)

# output 32 x 32 x 4 x 128
conv7 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(conv7)

# output of UpSampling2D(size=(2, 2, 2))(conv7): 64 x 64 x 8 x 128
# outputof up8: 64  x 64  x 8 x 192
up8 = merge([UpSampling3D(size=(2, 2, 2))(conv7), conv2], mode='concat', concat_axis=1)

# output 64 x 64 x 8 x 64
conv8 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(up8)

# output 64 x 64 x 8 x 64
conv8 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(conv8)

# output of UpSampling2D(size=(2, 2, 2))(conv8): 128 x 128 x 16 x 64
# outputof up9: 128 x 128 x 16 x 96
up9 = merge([UpSampling3D(size=(2, 2, 2))(conv8), conv1], mode='concat', concat_axis=1)

# output 128 x 128 x 16 x 32
conv9 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(up9)

# output 128 x 128 x 16 x 32
conv9 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(conv9)

# output 128 x 128 x 16 x 1
conv10 = Convolution3D(1, 1, 1, 1, activation='sigmoid')(conv9)

model = Model(input=inputs, output=conv10)

model.summary()


# In[ ]:

use_existing = False
model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])
model_checkpoint = ModelCheckpoint('unet3d_10.hdf5', monitor='loss', save_best_only=True)
#
# Should we load existing weights? 
# Set argument for call to train_and_predict to true at end of script
if use_existing:
    model.load_weights('./unet3d_10.hdf5')

# 
# The final results for this tutorial were produced using a multi-GPU
# machine using TitanX's.
# For a home GPU computation benchmark, on my home set up with a GTX970 
# I was able to run 20 epochs with a training set size of 320 and 
# batch size of 2 in about an hour. I started getting reseasonable masks 
# after about 3 hours of training. 
#
print('-'*30)
print('Fitting model...')
print('-'*30)
model.fit(imgs_train, imgs_mask_train, batch_size=2, nb_epoch=3, verbose=1, shuffle=True,
          callbacks=[model_checkpoint])



# In[9]:

# loading best weights from training session
print('-'*30)
print('Loading saved weights...')
print('-'*30)
model.load_weights('./unet3d_10.hdf5')

print('-'*30)
print('Predicting masks on test data...')
print('-'*30)
num_test = len(imgs_test)
imgs_mask_test = np.ndarray([num_test,1,16,128,128],dtype=np.float32)
for i in range(num_test):
    imgs_mask_test[i] = model.predict([imgs_test[i:i+1]], verbose=0)[0]
np.save('my_masksTestPredicted_16_128_128.npy', imgs_mask_test)
mean = 0.0
for i in range(num_test):
    mean+=dice_coef_np(imgs_mask_test_true[i,0], imgs_mask_test[i,0])
mean/=num_test
print("Mean Dice Coeff : ",mean)


# In[ ]:




# In[12]:

model.save(model_path+'model_unet3d.h5')


# In[ ]:



