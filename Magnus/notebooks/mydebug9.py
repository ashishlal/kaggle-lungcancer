
# coding: utf-8

# In[1]:

import os
os.environ['THEANO_FLAGS'] = 'floatX=float32,device=gpu0,nvcc.fastmath=True,lib.cnmem=0.85'
#THEANO_FLAGS=device=gpu python -c "import theano; print(theano.sandbox.cuda.device_properties(0))"
import keras
import theano

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


import matplotlib
matplotlib.use("Pdf")
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

#from helper_functions import *

from keras.layers import merge, Convolution2D, MaxPooling2D, Input, UpSampling2D
from keras.layers import merge, Convolution3D, MaxPooling3D, Input, UpSampling3D
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Model
from keras.optimizers import SGD
from keras import backend as K

K.set_image_dim_ordering('th')  # Theano dimension ordering in this code


# In[2]:

working_path = "/home/watts/lal/Kaggle/lung_cancer/cache/luna16/02242017_41_41_7/"
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


# In[19]:

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


# In[4]:

imgs_train = np.load(working_path+"my_train_images.npy").astype(np.float32)
imgs_mask_train = np.load(working_path+"my_train_masks.npy").astype(np.float32)

imgs_test = np.load(working_path+"my_test_images.npy").astype(np.float32)
imgs_mask_test_true = np.load(working_path+"my_test_masks.npy").astype(np.float32)
    
mean = np.mean(imgs_train)  # mean for data centering
std = np.std(imgs_train)  # std for data normalization

imgs_train -= mean  # images should already be standardized, but just in case
imgs_train /= std

train_x = imgs_train
train_y = imgs_mask_train

test_x = imgs_test
test_y = imgs_mask_test_true


# In[5]:

train_x = np.concatenate( [imgs_train[i] for i in range(imgs_train.shape[0])] )
train_y = np.concatenate([imgs_mask_train[i] for i in range(imgs_mask_train.shape[0])])
test_x = np.concatenate( [imgs_test[i] for i in range(imgs_test.shape[0])] )
test_y = np.concatenate([imgs_mask_test_true[i] for i in range(imgs_mask_test_true.shape[0])])


# In[6]:

train_x.shape


# In[7]:

train_y.shape


# In[8]:

train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1], train_x.shape[2],train_x.shape[3])

test_x = test_x.reshape(test_x.shape[0], 1, test_x.shape[1], test_x.shape[2],test_x.shape[3])


# In[9]:

train_y = np.array([train_y[i] for i in range(len(train_y))])

test_y = np.array([test_y[i] for i in range(len(test_y))])


# In[10]:

train_y = train_y.reshape(train_y.shape[0], 1, train_y.shape[1], train_y.shape[2],train_y.shape[3])


# In[11]:

train_y.shape


# In[12]:

train_x.shape


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


# In[20]:

inputs = Input((1,512, 512)) # width = 512, height = 512, depth = 1

# output W2 = (512 - 3 + 2*1)/1 + 1 = 512 x 512 x 32
# num output params = 3 * 3 * 1 * 32 + 32 = 320
conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs) 

# output 512 x 512 x 32
# num output params = 3 * 3 * 32 * 32 + 32 = 9248
conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)

# output = 256 x 256 x 32
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

# output = 256 x 256 x 64
conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)

# output = 256 x 256 x 64
conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)

# output = 128 x 128 x 64
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

# output = 128 x 128 x 128
conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)

# output = 128 x 128 x 128
conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)

# output = 64 x 64 x 128
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

# output = 64 x 64 x 256
conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)

# output = 64 x 64 x 256
conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)

# output = 32 x 32 x 256
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

# output = 32 x 32 x 512
conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)

# output = 32 x 32 x 512
conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

# output of UpSampling2D(size=(2, 2))(conv5): 64 x 64 x 512
# outputof up6: 64 x 64 x 768
up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)

# outputof up6: 64 x 64 x 256
conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)

# outputof up6: 64 x 64 x 256
conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

# output of UpSampling2D(size=(2, 2))(conv6): 128 x 128 x 256
# outputof up7: 128 x 128 x 384
up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)

# output: 128 x 128 x 128
conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)

# output: 128 x 128 x 128
conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

# output of UpSampling2D(size=(2, 2))(conv6): 256 x 256 x 256
# outputof up8: 256 x 256 x 192
up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)

# output: 256 x 256 x 64
conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)

# output: 256 x 256 x 64
conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

# output of UpSampling2D(size=(2, 2))(conv8):  512 x 512 x 64
# outputof up9: 512 x 512 x 96
up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)

# output: 512 x 512 x 32
conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)

# output: 512 x 512 x 32
conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

# output: 512 x 512 x 1
conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

model = Model(input=inputs, output=conv10)
model.summary()


# In[30]:

inputs = Input((1, 8, 40, 40))

# output W2 = (41 - 3 + 2*1)/1 + 1 = 40 x 40 x 8 x 32
conv1 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(inputs)

# output 40 x 40 x 8 x 32
conv1 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(conv1)

# output 20 x 20 x 4 x 32
pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

# output 20 x 20 x 4 x 64
conv2 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(pool1)

# output 20 x 20 x 4 x 64
conv2 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(conv2)

# output 10 x 10 x 2 x 64
pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

# output 10 x 10 x 1 x 128
#conv3 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(pool2)
#output 10 x 10 x 1 x 128
#conv3 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(conv3)
#pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

#conv4 = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same')(pool3)
#conv4 = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same')(conv4)
#pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

#conv5 = Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same')(pool4)
#conv5 = Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same')(conv5)

#up6 = merge([UpSampling3D(size=(2, 2, 2))(conv5), conv4], mode='concat', concat_axis=1)
#conv6 = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same')(up6)
#conv6 = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same')(conv6)

#up7 = merge([UpSampling3D(size=(2, 2, 2))(conv6), conv3], mode='concat', concat_axis=1)
#conv7 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(up7)
#conv7 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(conv7)

#up8 = merge([UpSampling3D(size=(2, 2, 2))(conv7), conv2], mode='concat', concat_axis=1)
#conv8 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(up8)
#conv8 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(conv8)

up9 = merge([UpSampling3D(size=(2, 2, 2))(conv2), conv1], mode='concat', concat_axis=1)
conv9 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(up9)
conv9 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(conv9)

conv10 = Convolution3D(1, 1, 1, 1, activation='sigmoid')(conv9)

model = Model(input=inputs, output=conv10)

model.summary()
#model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])


# In[15]:

input_img = Input(shape=(7, 1, 41, 41))
x = Convolution3D(nb_filter=40,kernel_dim1=3, kernel_dim2=3,kernel_dim3=3,border_mode='full',activation='relu')(input_img)

x = MaxPooling3D(pool_size=(5,5,5), strides=(5,5,5), border_mode='valid')(x)
x = Convolution3D(nb_filter=20,kernel_dim1=3, kernel_dim2=3,kernel_dim3=3,border_mode='full',activation='relu')(x)
x = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), border_mode='valid')(x)
#x = LSTM(10, input_shape=(7,1,41,41))(x)
x = Dense(60, activation='relu', name="hidden_one")(x)
x = Dropout(.5)(x)
x = Dense(10, activation='relu', name="hidden_two")(x)
predictions = Dense(1, activation='sigmoid', name="output_one")(x)
model = Model(input=input_img, output=predictions)
sgd = SGD(lr=0.05, decay=1e-6, momentum=0.5, nesterov=True)
model.compile(optimizer=sgd, loss=dice_coef_loss, metrics=[dice_coef])
#model.summary()

get_1st_layer_output = K.function([model.layers[0].input],
                                  [model.layers[1].output])
layer_output = get_1st_layer_output(train_x)[0]

layer_output
#model.fit(train_x, train_y,nb_epoch=25)


# In[29]:

inputs = Input((1, 7, 41, 41))

# output  41 x 41 x 7 x 50
conv1 = Convolution3D(50, 7, 16, 16, activation='relu', border_mode='same')(inputs)

# output 41 x 41 x 7 x 50
pool1 = MaxPooling3D(pool_size=(1, 1, 1))(conv1)

# output 41 x 41 x 7 x 50
conv2 = Convolution3D(50, 50, 7, 7, activation='relu', border_mode='same')(pool1)

# output 20 x 20 x 3 x 64
#conv2 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(conv2)

# output 20 x 20 x 3 x 50
pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

# output 41 x 41 x 7 x 50
conv3 = Convolution3D(80, 50, 3, 3, activation='relu', border_mode='same')(pool2)

# output 41 x 41 x 7 x 50
conv4 = Convolution3D(100, 80, 2, 2, activation='relu', border_mode='same')(conv3)

# output 10 x 10 x 1 x 128
#conv3 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(pool2)
#output 10 x 10 x 1 x 128
#conv3 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(conv3)
#pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

#conv4 = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same')(pool3)
#conv4 = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same')(conv4)
#pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

#conv5 = Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same')(pool4)
#conv5 = Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same')(conv5)

#up6 = merge([UpSampling3D(size=(2, 2, 2))(conv5), conv4], mode='concat', concat_axis=1)
#conv6 = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same')(up6)
#conv6 = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same')(conv6)

#up7 = merge([UpSampling3D(size=(2, 2, 2))(conv6), conv3], mode='concat', concat_axis=1)
#conv7 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(up7)
#conv7 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(conv7)

#up8 = merge([UpSampling3D(size=(2, 2, 2))(conv7), conv2], mode='concat', concat_axis=1)
#conv8 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(up8)
#conv8 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(conv8)

#up9 = merge([UpSampling3D(size=(2, 2, 2))(conv2), conv1], mode='concat', concat_axis=1)
#conv9 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(up9)
#conv9 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(conv9)

conv10 = Convolution3D(1, 1, 1, 1, activation='sigmoid')(conv4)

model = Model(input=inputs, output=conv10)

model.summary()
#model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])


# In[ ]:



