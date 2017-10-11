
# coding: utf-8

# In[1]:

import numpy as np
import pickle
import pandas as pd
from skimage import measure
import datetime
import os
from tqdm import tqdm
from time import strftime
# from sklearn import cross_validation
# from sklearn.cross_validation import StratifiedKFold as KFold
# from sklearn.metrics import classification_report
# from sklearn.ensemble import RandomForestClassifier as RF
# import xgboost as xgb


# In[2]:

nodule_path = "/home/watts/lal/Kaggle/lung_cancer/cache/predictions/"
working_path = "/home/watts/lal/Kaggle/lung_cancer/"


# In[3]:

def check_if_image_exists(fname):
    fname = os.path.join(working_path+'data/stage1/stage1/', fname)
    return os.path.exists(fname)

def check_if_scan_exists(folder):
    folder = os.path.join(working_path+'data/stage1/stage1/', folder)
    return os.path.isdir(folder)

def check_if_scan_exists2(folder):
    folder = os.path.join(working_path+'data/stage2/', folder)
    return os.path.isdir(folder)

def get_current_date():
    return strftime('%Y%m%d')


# In[4]:

num_slices = 16
img_width = 128
img_height = 128


# In[5]:

def getRegionFromMap(slice_npy):
    thr = np.where(slice_npy > np.mean(slice_npy),0.,1.0)
    # print thr
    label_image = measure.label(thr)
    labels = label_image.astype(int)
    regions = measure.regionprops(labels)
    return regions

def getRegionMetricRow(fname = "nodules.npy"):
    # fname, numpy array of dimension [#slices, 1, 512, 512] containing the images
    seg = np.load(fname)
    nslices = seg.shape[0]

    #metrics
    totalArea = 0.
    avgArea = 0.
    maxArea = 0.
    avgEcc = 0.
    avgEquivlentDiameter = 0.
    stdEquivlentDiameter = 0.
    weightedX = 0.
    weightedY = 0.
    numNodes = 0.
    numNodesperSlice = 0.
    # crude hueristic to filter some bad segmentaitons
    # do not allow any nodes to be larger than 10% of the pixels to eliminate background regions
    maxAllowedArea = 0.10 * 512 * 512

    areas = []

    eqDiameters = []
    for slicen in range(nslices):
        regions = getRegionFromMap(seg[slicen,0,:,:])
        for region in regions:
            if region.area > maxAllowedArea:
                continue
            totalArea += region.area
            areas.append(region.area)
            avgEcc += region.eccentricity
            avgEquivlentDiameter += region.equivalent_diameter
            eqDiameters.append(region.equivalent_diameter)
            weightedX += region.centroid[0]*region.area
            weightedY += region.centroid[1]*region.area
            numNodes += 1

    weightedX = weightedX / totalArea
    weightedY = weightedY / totalArea
    avgArea = totalArea / numNodes

    avgEcc = avgEcc / numNodes
    avgEquivlentDiameter = avgEquivlentDiameter / numNodes
    stdEquivlentDiameter = np.std(eqDiameters)

    maxArea = max(areas)


    numNodesperSlice = numNodes*1. / nslices


    return np.array([avgArea,maxArea,avgEcc,avgEquivlentDiameter,                     stdEquivlentDiameter, weightedX, weightedY, numNodes, numNodesperSlice])

def getRegionMetricRow2(fname = "nodules.npy"):
    # fname, numpy array of dimension [1, #slices, 128, 128] containing the images
    seg = np.load(fname)
    nslices = seg.shape[1]

    #print nslices
    #metrics
    totalArea = 0.
    avgArea = 0.
    maxArea = 0.
    avgEcc = 0.
    avgEquivlentDiameter = 0.
    stdEquivlentDiameter = 0.
    weightedX = 0.
    weightedY = 0.
    numNodes = 0.
    numNodesperSlice = 0.
    # crude hueristic to filter some bad segmentaitons
    # do not allow any nodes to be larger than 10% of the pixels to eliminate background regions
    #maxAllowedArea = 0.10 * 128 * 128
    maxAllowedArea = img_width * img_height

    areas = []

    eqDiameters = []
    for slicen in range(nslices):
        #regions = getRegionFromMap(seg[slicen,0,:,:])
        regions = getRegionFromMap(seg[0,slicen,:,:])
        for region in regions:
            if region.area > maxAllowedArea:
                #print region.area, maxAllowedArea
                continue
            
            
            totalArea += region.area
            areas.append(region.area)
            avgEcc += region.eccentricity
            avgEquivlentDiameter += region.equivalent_diameter
            eqDiameters.append(region.equivalent_diameter)
            weightedX += region.centroid[0]*region.area
            weightedY += region.centroid[1]*region.area
            numNodes += 1

    weightedX = weightedX / totalArea
    weightedY = weightedY / totalArea
    avgArea = totalArea / numNodes

    avgEcc = avgEcc / numNodes
    avgEquivlentDiameter = avgEquivlentDiameter / numNodes
    stdEquivlentDiameter = np.std(eqDiameters)

    maxArea = max(areas)


    numNodesperSlice = numNodes*1. / nslices


    return avgArea,maxArea,avgEcc,avgEquivlentDiameter,                     stdEquivlentDiameter, weightedX, weightedY, numNodes, numNodesperSlice



# In[6]:

def createFeatureDataset(nodfiles=None):
    if nodfiles == None:
        # directory of numpy arrays containing masks for nodules
        # found via unet segmentation
        noddir = "/training_set/"
        nodfiles = glob(noddir +"*npy")
    # dict with mapping between training examples and true labels
    # the training set is the output masks from the unet segmentation
    truthdata = pickle.load(open("truthdict.pkl",'r'))
    numfeatures = 9
    feature_array = np.zeros((len(nodfiles),numfeatures))
    truth_metric = np.zeros((len(nodfiles)))

    for i,nodfile in enumerate(nodfiles):
        patID = nodfile.split("_")[2]
        truth_metric[i] = truthdata[int(patID)]
        feature_array[i] = getRegionMetricRow(nodfile)

    np.save("dataY.npy", truth_metric)
    np.save("dataX.npy", feature_array)

def createFeatureDataset2(nodfiles=None):
    if nodfiles == None:
        # directory of numpy arrays containing masks for nodules
        # found via unet segmentation
        noddir = "/training_set/"
        nodfiles = glob(noddir +"*npy")
    # dict with mapping between training examples and true labels
    # the training set is the output masks from the unet segmentation
    truthdata = pickle.load(open("truthdict.pkl",'r'))
    numfeatures = 9
    feature_array = np.zeros((len(nodfiles),numfeatures))
    truth_metric = np.zeros((len(nodfiles)))

    for i,nodfile in enumerate(nodfiles):
        patID = nodfile.split("_")[2]
        truth_metric[i] = truthdata[int(patID)]
        feature_array[i] = getRegionMetricRow(nodfile)

    np.save("dataY.npy", truth_metric)
    np.save("dataX.npy", feature_array)


# In[7]:

df = pd.read_csv(working_path+'data/stage1/stage1_labels_all.csv')

df['scan_folder'] = df['id']

df['exist'] = df['scan_folder'].apply(check_if_scan_exists)

print '%i does not exists' % (len(df) - df['exist'].sum())
print df[~df['exist']]

df = df[df['exist']]
df = df.reset_index(drop=True)


# In[8]:

data = []
IMG_PX_SIZE = img_width
IMG_PX_SIZE_ORG = 512
HM_SLICES = num_slices
for i, row in tqdm(df.iterrows(), total=len(df)):
#     if i != 0:
#         continue
    scan_folder = row['scan_folder']
    # X_nodule_fname = nodule_path+'X_nodule_%s_%s_%s_%s.npy' % (scan_folder, HM_SLICES, IMG_PX_SIZE, IMG_PX_SIZE)
    X_nodule_fname = nodule_path+'stage1/%s.npy' % scan_folder
    avgArea,maxArea,avgEcc,avgEquivlentDiameter,                     stdEquivlentDiameter, weightedX, weightedY, numNodes, numNodesperSlice     = getRegionMetricRow2(X_nodule_fname)

    cancer = row['cancer']
    t = {'scan_folder': scan_folder,
         'avgArea': avgArea, 
         'maxArea':maxArea, 
         'avgEcc': avgEcc, 
         'avgEquivlentDiameter': avgEquivlentDiameter,
         'stdEquivlentDiameter': stdEquivlentDiameter,
         'weightedX': weightedX,
         'weightedY': weightedY,
         'numNodes': numNodes,
         'numNodesperSlice': numNodesperSlice,
         'output': cancer
        }
    data.append(t)
df = pd.DataFrame(data)
train_fname = working_path+'cache/my_train_%d_%d_%d_%s.csv' % (num_slices, img_width, img_height, get_current_date())
df.to_csv(train_fname, sep=',', index_label = 'id')
print 'Done'
now = datetime.datetime.now()
print now


# In[10]:

df.head(20)


# In[11]:

df = pd.read_csv(working_path+'data/stage2_sample_submission.csv')

df['scan_folder'] = df['id']

df['exist'] = df['scan_folder'].apply(check_if_scan_exists2)

print '%i does not exists' % (len(df) - df['exist'].sum())
print df[~df['exist']]

df = df[df['exist']]
df = df.reset_index(drop=True)


# In[12]:

data = []
IMG_PX_SIZE = img_width
IMG_PX_SIZE_ORG = 512
HM_SLICES = num_slices
for i, row in tqdm(df.iterrows(), total=len(df)):
#     if i != 0:
#         continue
    scan_folder = row['scan_folder']
    #X_nodule_fname = nodule_path+'X_test_nodule_%s_%s_%s_%s.npy' % (scan_folder, HM_SLICES, IMG_PX_SIZE, IMG_PX_SIZE)
    X_nodule_fname = nodule_path+'stage2/%s.npy' % scan_folder
    avgArea,maxArea,avgEcc,avgEquivlentDiameter,                     stdEquivlentDiameter, weightedX, weightedY, numNodes, numNodesperSlice     = getRegionMetricRow2(X_nodule_fname)

    #cancer = row['cancer']
    t = {'scan_folder': scan_folder,
         'avgArea': avgArea, 
         'maxArea':maxArea, 
         'avgEcc': avgEcc, 
         'avgEquivlentDiameter': avgEquivlentDiameter,
         'stdEquivlentDiameter': stdEquivlentDiameter,
         'weightedX': weightedX,
         'weightedY': weightedY,
         'numNodes': numNodes,
         'numNodesperSlice': numNodesperSlice
        }
    data.append(t)
df = pd.DataFrame(data)
test_fname = working_path+'cache/my_test_%d_%d_%d_%s.csv' % (num_slices, img_width, img_height, get_current_date())
df.to_csv(test_fname, sep=',', index_label = 'id')
print 'Done'
now = datetime.datetime.now()
print now


# In[13]:

df.head(20)


# In[ ]:




# In[ ]:



