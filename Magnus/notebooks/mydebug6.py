
# coding: utf-8

# In[1]:

import os
os.environ['THEANO_FLAGS'] = 'floatX=float32,device=gpu0,nvcc.fastmath=True,lib.cnmem=0.85'
#THEANO_FLAGS=device=gpu python -c "import theano; print(theano.sandbox.cuda.device_properties(0))"
import lasagne
import theano

import sklearn
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, f1_score, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn import metrics

from lasagne import layers
from lasagne.nonlinearities import  sigmoid, softmax, rectify, tanh, linear
from lasagne.updates import nesterov_momentum, adagrad, adam
from nolearn.lasagne import NeuralNet
import lasagne.layers.dnn

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

working_path = "/home/watts/lal/Kaggle/lung_cancer/cache/luna16/02242017_41_41_7/"


# In[224]:

#Load and reshape data
#train_x = pickle.load(open("/home/ubuntu/41_41_7_train_test/train_x.p","rb"))
#train_y = pickle.load(open("/home/ubuntu/41_41_7_train_test/train_y.p","rb"))
#test_x = pickle.load(open("/home/ubuntu/41_41_7_train_test/test_x.p","rb"))
#test_y = pickle.load(open("/home/ubuntu/41_41_7_train_test/test_y.p","rb"))

#train_x = train_x.astype(np.float32)
#train_y = train_y.astype(np.int32)
#test_x = test_x.astype(np.float32)
#test_y = test_y.astype(np.int32)

#train_x = train_x.reshape(train_x.shape[0],1,train_x.shape[1],train_x.shape[2],train_x.shape[3])
#test_x = test_x.reshape(test_x.shape[0],1,test_x.shape[1],test_x.shape[2],test_x.shape[3])

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



# In[225]:

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_ct_scan(scan):
    f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(25, 25))
    for i in range(0, scan.shape[0], 5):
        plots[int(i / 20), int((i % 20) / 5)].axis('off')
        plots[int(i / 20), int((i % 20) / 5)].imshow(scan[i], cmap=plt.cm.bone) 
        
def plot_ct_scan_my(scan):
    f, plots = plt.subplots(int(scan.shape[1] / 3) + 1, 4, figsize=(3, 3))
    for i in range(0, scan.shape[1], 5):
        plots[int(i / 3), int((i % 3) / 5)].axis('off')
        plots[int(i / 3), int((i % 3) / 5)].imshow(scan[i], cmap=plt.cm.bone) 
def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    #p =image
    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


# In[226]:

imgs_train[0].shape


# In[ ]:




# In[ ]:




# In[227]:

#train_y = []
#test_y = []
train_x = np.concatenate( [imgs_train[i] for i in range(imgs_train.shape[0])] )
train_y = np.array([imgs_mask_train[i] for i in range(imgs_mask_train.shape[0])])
test_x = np.concatenate( [imgs_test[i] for i in range(imgs_test.shape[0])] )
test_y = np.array([imgs_mask_test_true[i] for i in range(imgs_mask_test_true.shape[0])])


# In[228]:

train_x.shape


# In[229]:

train_y[0].shape


# In[230]:

test_x.shape


# In[231]:

test_y.shape


# In[232]:

train_x = train_x.reshape(train_x.shape[0],1,train_x.shape[1],train_x.shape[2],train_x.shape[3])

test_x = test_x.reshape(test_x.shape[0],1,test_x.shape[1],test_x.shape[2],test_x.shape[3])


# In[233]:

train_y = train_y.reshape(train_y.shape[0], -1)
test_y = test_y.reshape(test_y.shape[0], -1)


# In[234]:

train_y = [train_y[i] for i in range(train_y.shape[0])]
test_y = [test_y[i] for i in range(test_y.shape[0])]


# In[240]:

from sklearn import preprocessing
LE = preprocessing.LabelEncoder()
#for i in range(train_x.shape[0]):
train_y = LE.fit_transform(train_y)
for i in range(test_x.shape[0]):
    test_y[i] = LE.fit_transform(test_y[i])


# In[236]:

train_x.shape


# In[237]:

train_y = np.array([train_y[i] for i in range(len(train_y))])

test_y = np.array([test_y[i] for i in range(len(test_y))])


# In[238]:

train_y.shape


# In[239]:

train_y = np.reshape(train_y,[train_y.shape[0],])
test_y = np.reshape(test_y,[test_y.shape[0],])


# In[220]:

nn = NeuralNet(
            layers=[('input', layers.InputLayer),
                   ('conv1', layers.dnn.Conv3DDNNLayer),
                    ('pool1', layers.dnn.MaxPool3DDNNLayer),
                   ('conv2', layers.dnn.Conv3DDNNLayer),
                    ('pool2', layers.dnn.MaxPool3DDNNLayer),
                   ('lstm1', layers.LSTMLayer),
                    ('hidden1', layers.DenseLayer),
                   ('dropout1', layers.DropoutLayer),
                    ('hidden2', layers.DenseLayer),
                    ('output', layers.DenseLayer),
            ],
         
            # Input Layer
            input_shape=(None,1, 7, 41, 41),
            #input_shape=(None,1, 7 * 41 * 41),
        
            #Convolutional1 Layer
         
            conv1_num_filters = 40,
            conv1_nonlinearity = rectify,
            conv1_filter_size=(3, 3, 3),
            conv1_stride = (1,1,1),
            conv1_pad = "full",
        
            # Pooling1 Layer
            pool1_pool_size = 5,
        
            #Convolutional Layer 2
            conv2_num_filters = 20,
            conv2_nonlinearity = rectify,
            conv2_filter_size=(3, 3,3),
            conv2_pad = "full",
        
            #Pooling2 Layer
            pool2_pool_size = 2,
        
             #LSTM Layer
            lstm1_num_units = 10,
        
            #1st Hidden Layer
            hidden1_num_units=60,
            hidden1_nonlinearity=rectify,
        
            #Dropout Layer
            dropout1_p = 0.5,
        
            # 2nd Hidden Layer
            hidden2_num_units=10,
            hidden2_nonlinearity=rectify,
        
            # Output Layer
            output_num_units=2,
            output_nonlinearity=softmax,
        
            # Optimization
            update=nesterov_momentum,
            update_learning_rate=0.05,
            update_momentum=0.5,
            max_epochs=25,
        
            #update = adagrad,
            #update_learning_rate = .07,
            #max_epochs = 50,
        
            # Others
            regression=False,
            verbose=1,
     )



# In[221]:

nn.fit(train_x, train_y)
predict_y = nn.predict(test_x)

print "Accuracy Score: " +str(accuracy_score(test_y, predict_y))
print "Precision Score: " + str(precision_score(test_y, predict_y))
print "Recall Score: " + str(recall_score(test_y, predict_y))
print "F1 Score: " + str(f1_score(test_y, predict_y))


# In[ ]:



