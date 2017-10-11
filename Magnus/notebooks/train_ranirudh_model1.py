
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
import argparse
import json as simplejson
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

num_slices = 16
img_width = 128
img_height = 128

validate_model_weights = 'unet3d_%d_validate.hdf5' % img_width
validate_model = 'model_validate_unet3d_%d.h5' % img_width

train_model_weights = 'unet3d_%d_train.hdf5' % img_width
train_model = 'model_train_unet3d_%d.h5' % img_width

model_arch = 'model_unet3d_%d.json' % img_width
model_weights = 'model_unet3d_%d.h5' % img_width


# In[5]:

def get_validation_data():
    my_train_imgs_fname = 'my_train_images_%d_%d_%d.npy' % (num_slices, img_width, img_height)
    my_train_masks_fname = 'my_train_masks_%d_%d_%d.npy' % (num_slices, img_width, img_height)
    my_test_imgs_fname = 'my_test_images_%d_%d_%d.npy' % (num_slices, img_width, img_height)
    my_test_masks_fname = 'my_test_masks_%d_%d_%d.npy' % (num_slices, img_width, img_height)

    imgs_train = np.load(working_path+my_train_imgs_fname).astype(np.float32)
    imgs_mask_train = np.load(working_path+my_train_masks_fname).astype(np.float32)

    imgs_test = np.load(working_path+my_test_imgs_fname).astype(np.float32)
    imgs_mask_test_true = np.load(working_path+my_test_masks_fname).astype(np.float32)

    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean  # images should already be standardized, but just in case
    imgs_train /= std

    train_x = imgs_train
    train_y = imgs_mask_train

    test_x = imgs_test
    test_y = imgs_mask_test_true
    
    print train_x.shape
    print train_y.shape
    
    return train_x, train_y, test_x, test_y


# In[30]:

#train_x = np.concatenate( [imgs_train[i] for i in range(imgs_train.shape[0])] )
#train_y = np.concatenate([imgs_mask_train[i] for i in range(imgs_mask_train.shape[0])])
#est_x = np.concatenate( [imgs_test[i] for i in range(imgs_test.shape[0])] )
#test_y = np.concatenate([imgs_mask_test_true[i] for i in range(imgs_mask_test_true.shape[0])])


# In[33]:

#train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1], train_x.shape[2],train_x.shape[3])

#test_x = test_x.reshape(test_x.shape[0], 1, test_x.shape[1], test_x.shape[2],test_x.shape[3])


# In[34]:

#train_y = np.array([train_y[i] for i in range(len(train_y))])

#test_y = np.array([test_y[i] for i in range(len(test_y))])


# In[35]:

#train_y = train_y.reshape(train_y.shape[0], 1, train_y.shape[1], train_y.shape[2],train_y.shape[3])
#test_y = test_y.reshape(test_y.shape[0], 1, test_y.shape[1], test_y.shape[2],test_y.shape[3])


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


# In[16]:


num_slices = 7
img_width = 41
img_height = 41
#def get_unet3d_gen_model():
inputs = Input((1, num_slices, img_height, img_width))

# output W2 = 128 x 128 x 16 x 50
conv1 = Convolution3D(50, 16, 16, 7, activation='relu', border_mode='same')(inputs)

# output 128 x 128 x 8 x 50
pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

# output 128 x 128 x 8 x 64
conv2 = Convolution3D(50, 7, 7, 50, activation='relu', border_mode='same')(pool1)

# output 64 x 64 x 4 x 64
pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

# output 64 x 64 x 4 x 128
conv3 = Convolution3D(80, 3, 3, 50, activation='relu', border_mode='same')(pool2)

#output 64 x 64 x 4 x 128
conv3 = Convolution3D(100, 2, 2, 80, activation='relu', border_mode='same')(conv3)

# output 256 x 256 x 16 x 1
conv10 = Dense(1, activation='sigmoid')(conv3)
conv11 = Dense(1, activation='sigmoid')(conv10)

model = Model(input=inputs, output=conv11)

model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])

model.summary()

#return model


# In[ ]:




# In[ ]:

def train_unet3d_model_for_validation(model, train_x, train_y, use_existing = False):
    model_checkpoint = ModelCheckpoint(model_path+validate_model_weights, monitor='loss', save_best_only=True)
    
    #
    # Should we load existing weights? 
    # Set argument for call to train_and_predict to true at end of script
    if use_existing:
        model.load_weights(model_path+validate_model_weights)

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
    nb_epoch = 3

    model.fit(train_x, train_y, batch_size=2, nb_epoch=nb_epoch, verbose=2, shuffle=True,
              callbacks=[model_checkpoint])

    return model


# In[ ]:




# In[9]:

def predict_for_validation(model, test_x, test_y, use_existing = False):
    # loading best weights from training session
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights(model_path+validate_model_weights)

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    num_test = len(test_x)
    imgs_mask_test = np.ndarray([num_test,1,num_slices,img_height,img_width],dtype=np.float32)
    for i in range(num_test):
        imgs_mask_test[i] = model.predict([test_x[i:i+1]], verbose=0)[0]
    #np.save('my_masksTestPredicted_16_128_128.npy', imgs_mask_test)
    mean = 0.0
    for i in range(num_test):
        mean+=dice_coef_np(test_y[i,0], imgs_mask_test[i,0])
    mean/=num_test
    print("Mean Dice Coeff : ",mean)


# In[ ]:

def save_unet3d_validation(model):
    model.save(model_path+validate_model)


# In[ ]:

def get_training_data():
    my_train_img_fname = 'my_image_%d_%d_%d.npy' % (num_slices, img_width, img_height)
    my_train_mask_fname = 'my_mask_%d_%d_%d.npy' % (num_slices, img_width, img_height)


    my_imgs_train = np.load(working_path+my_train_img_fname).astype(np.float32)
    my_imgs_mask_train = np.load(working_path+my_train_mask_fname).astype(np.float32)


    mean = np.mean(my_imgs_train)  # mean for data centering
    std = np.std(my_imgs_train)  # std for data normalization

    my_imgs_train -= mean  # images should already be standardized, but just in case
    my_imgs_train /= std

    train_x = my_imgs_train
    train_y = my_imgs_mask_train
    return train_x, train_y


# In[ ]:

def train_unet3d(model, train_x, train_y):
    model_checkpoint = ModelCheckpoint(model_path+train_model_weights, monitor='loss', verbose = 1, save_best_only=True)
    print('-'*30)
    print('Fitting train model...')
    print('-'*30)
    model.fit(train_x, train_y, batch_size=2, nb_epoch=3, verbose=2, shuffle=True,
              callbacks=[model_checkpoint])
    return model


# In[ ]:




# In[ ]:

def save_unet3d(model):
    model.save(model_path+train_model)
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_path+model_arch, "w") as json_file:
        json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))

    # serialize weights to HDF5
    model.save_weights(model_path+model_weights)
    print("Saved model to disk")


# In[ ]:

def do_validation_and_training():
    train_x, train_y, test_x, test_y = get_validation_data()
    model = get_unet3d_gen_model()
    model = train_unet3d_model_for_validation(model, train_x, train_y)
    predict_for_validation(model, test_x, test_y)
    save_unet3d_validation(model)
    train_x, train_y = get_training_data()
    model = train_unet3d(model, train_x, train_y)
    save_unet3d(model)


# In[ ]:

def do_training():
    train_x, train_y = get_training_data()
    model = get_unet3d_gen_model()
    model = train_unet3d(model, train_x, train_y)
    save_unet3d(model)


# In[ ]:

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', required=True, type=int, help='Size of the input image')
    parser.add_argument('--nslices', required=True, type=int, help='Number of slices')
    parser.add_argument('--overwrite', action='store_true', help='Overwirte existing cache')
    parser.add_argument('--validation', dest='validation', action='store_true', help='Do validation')
    parser.add_argument('--no-validation', dest='validation', action='store_false', help='Do validation')

    args = parser.parse_args()

    img_width = args.size
    img_height = args.size
    num_slices = args.nslices
    
    validate_model_weights = 'unet3d_%d_validate.hdf5' % img_width
    validate_model = 'model_validate_unet3d_%d.h5' % img_width

    train_model_weights = 'unet3d_%d_train.hdf5' % img_width
    train_model = 'model_train_unet3d_%d.h5' % img_width

    model_arch = 'model_unet3d_%d.json' % img_width
    model_weights = 'model_unet3d_%d.h5' % img_width

    if args.validation:
        print 'doing validation..'
        do_validation_and_training()
    else:
        print 'doing training..'
        do_training()
    
    


# In[ ]:



