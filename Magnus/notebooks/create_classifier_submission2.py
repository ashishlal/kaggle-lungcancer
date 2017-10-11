
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn import cross_validation, metrics
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier as RF
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
#from sklearn.grid_search import GridSearchCV   #Perforing grid search
#import matplotlib.pylab as plt
#%matplotlib inline
from tqdm import tqdm
import datetime
from time import strftime

from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


# In[2]:

working_path = "/home/watts/lal/Kaggle/lung_cancer/"


# In[3]:

def get_current_date():
    return strftime('%Y%m%d')


# In[4]:

num_slices = 16
img_width = 128
img_height = 128


# In[5]:

train_fname = 'cache/my_train_%d_%d_%d_%s.csv' % (num_slices, img_width, img_height, get_current_date())
test_fname = 'cache/my_test_%d_%d_%d_%s.csv' % (num_slices, img_width, img_height, get_current_date())
train = pd.read_csv(working_path+train_fname, sep=',')
test = pd.read_csv(working_path+test_fname, sep=',')
target = 'output'
idcol = 'id'
scan_folder = 'scan_folder'


# In[19]:

df = train
df = df.drop('output',axis=1)
df = df.drop('id',axis=1)
df = df.drop('scan_folder',axis=1)

X = df
y = train['output']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234)


def objective(space):

    clf = xgb.XGBClassifier(learning_rate=0.325,
                            silent=True,
                            objective="binary:logistic",
                            nthread=-1,
                            gamma=0.85,
                            min_child_weight=5,
                            max_delta_step=1,
                            subsample=0.85,
                            colsample_bytree=0.55,
                            colsample_bylevel=1,
                            reg_alpha=0.5,
                            reg_lambda=1,
                            scale_pos_weight=1,
                            base_score=0.5,
                            seed=0,
                            missing=None,
                            n_estimators=1920, max_depth=6)

    
    eval_set  = [( X_train, y_train), (X_test, y_test)]

    clf.fit(X_train, y_train,
            eval_set=eval_set, eval_metric="logloss", 
            early_stopping_rounds=100)

    pred = clf.predict_proba(X_test)[:,1]
    loss = log_loss(y_test, pred)
    print "logloss:", loss

    return{'loss':loss, 'status': STATUS_OK }



# In[20]:

space = {
        'max_depth': hp.quniform('max_depth', 1, 13, 1),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
        'learning_rate': hp.quniform('learning_rate', 0.025, 0.5, 0.025),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
        'n_estimators': hp.quniform('n_estimators', 100, 1000, 5),
        'silent' : 1
}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

print best


# In[19]:

# df = train
# df = df.drop('output',axis=1)
# df = df.drop('id',axis=1)
# df = df.drop('scan_folder',axis=1)

# X = df
# y = train['output']

# df1 = test
# df1 = df1.drop('id',axis=1)
# df1 = df1.drop('scan_folder',axis=1)
# X_test = df1

# clf = xgb.XGBClassifier(learning_rate=0.275,
#                         silent=True,
#                         objective="binary:logistic",
#                         nthread=-1,
#                         gamma=0.85,
#                         min_child_weight=5,
#                         max_delta_step=1,
#                         subsample=0.85,
#                         colsample_bytree=0.7,
#                         colsample_bylevel=1,
#                         reg_alpha=0.5,
#                         reg_lambda=1,
#                         scale_pos_weight=1,
#                         base_score=0.5,
#                         seed=0,
#                         missing=None,
#                         n_estimators=360, max_depth=4)


# eval_set  = [( X, y)]

# clf.fit(X, y,
#         eval_set=eval_set, eval_metric="logloss", 
#         early_stopping_rounds=100)


# #loss = log_loss(y_test, pred)
# #print "logloss:", loss

# #return{'loss':loss, 'status': STATUS_OK }



# In[25]:

# Y_pred = clf.predict_proba(X_test)[:,1]


# In[26]:

# print Y_pred


# In[6]:

df = train
df = df.drop('output',axis=1)
df = df.drop('id',axis=1)
df = df.drop('scan_folder',axis=1)

X_train = df
Y_train = train['output']
T_train_xgb = xgb.DMatrix(X_train, Y_train)

params = {
    'learning_rate':0.275,
    'objective':"binary:logistic",
    'nthread':-1,
    'gamma':0.85,
    'min_child_weight':5,
    'max_delta_step':1,
    'subsample':0.85,
    'colsample_bytree':0.70,
    'colsample_bylevel':1,
    'reg_alpha':0.5,
    'reg_lambda':1,
    'scale_pos_weight':1,
    'base_score':0.5,
    'seed':0,
    'missing':None,
    'n_estimators':360, 
    'max_depth':4}



# In[7]:

xgb.cv(params = params, dtrain = T_train_xgb, num_boost_round = 3000, nfold = 10,
                metrics = ['logloss'], # Make sure you enter metrics inside a list or you may encounter issues!
                early_stopping_rounds = 100) 


# In[8]:

bst = xgb.train(dtrain=T_train_xgb,params=params, num_boost_round=24)


# In[9]:

df = test
df = df.drop('id',axis=1)
df = df.drop('scan_folder',axis=1)

X_test = df


# In[10]:

Y_pred = bst.predict(xgb.DMatrix(X_test))
print Y_pred


# In[11]:

import string
data = []
cols = ['id', 'cancer']
df = test
for i, row in tqdm(df.iterrows(), total=len(df)):
    scan_folder = row['scan_folder']
    cancer = Y_pred[i]
    t = {
         'id': scan_folder,
         'cancer': cancer
        }
    data.append(t)
df_sub = pd.DataFrame(data)
df_sub = df_sub[cols]
now = str(datetime.datetime.now())
now = now.replace(' ','-')
now = now.replace(':','-')
print now
sub_fname = working_path+'cache/submissions/my_sub_%s.csv' % now
df_sub.to_csv(sub_fname, sep=',', index=False)
print 'Done'
print sub_fname


# In[44]:

df0 = train

df1 = test
df1['output'] = Y_pred

df = pd.concat([df0,df1])
df2 = df

df = df.drop('output',axis=1)
df = df.drop('id',axis=1)
df = df.drop('scan_folder',axis=1)

X_train = df
Y_train = df2['output']
T_train_xgb = xgb.DMatrix(X_train, Y_train)

params = {
    'learning_rate':0.275,
    'objective':"binary:logistic",
    'nthread':-1,
    'gamma':0.85,
    'min_child_weight':5,
    'max_delta_step':1,
    'subsample':0.85,
    'colsample_bytree':0.70,
    'colsample_bylevel':1,
    'reg_alpha':0.5,
    'reg_lambda':1,
    'scale_pos_weight':1,
    'base_score':0.5,
    'seed':0,
    'missing':None,
    'n_estimators':360, 
    'max_depth':4}



# In[45]:

xgb.cv(params = params, dtrain = T_train_xgb, num_boost_round = 3000, nfold = 10,
                metrics = ['logloss'], # Make sure you enter metrics inside a list or you may encounter issues!
                early_stopping_rounds = 100) 


# In[46]:

bst = xgb.train(dtrain=T_train_xgb,params=params, num_boost_round=12)


# In[49]:

df = test
df = df.drop('id',axis=1)
df = df.drop('scan_folder',axis=1)
df = df.drop('output',axis=1)

X_test = df


# In[50]:

Y_pred = bst.predict(xgb.DMatrix(X_test))
print Y_pred


# In[51]:

import string
data = []
cols = ['id', 'cancer']
df = test
for i, row in tqdm(df.iterrows(), total=len(df)):
    scan_folder = row['scan_folder']
    cancer = Y_pred[i]
    t = {
         'id': scan_folder,
         'cancer': cancer
        }
    data.append(t)
df_sub = pd.DataFrame(data)
df_sub = df_sub[cols]
now = str(datetime.datetime.now())
now = now.replace(' ','-')
now = now.replace(':','-')
print now
sub_fname = working_path+'cache/submissions/my_sub_%s.csv' % now
df_sub.to_csv(sub_fname, sep=',', index=False)
print 'Done'
print sub_fname


# In[ ]:



