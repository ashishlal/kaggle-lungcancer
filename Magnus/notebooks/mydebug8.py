
# coding: utf-8

# In[1]:

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

import sys
import os
import gzip
import pickle
import numpy



# In[2]:


PY2 = sys.version_info[0] == 2

if PY2:
    from urllib import urlretrieve

    def pickle_load(f, encoding):
        return pickle.load(f)
else:
    from urllib.request import urlretrieve

    def pickle_load(f, encoding):
        return pickle.load(f, encoding=encoding)

DATA_URL = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
DATA_FILENAME = 'mnist.pkl.gz'



# In[6]:


def _load_data(url=DATA_URL, filename=DATA_FILENAME):
    """Load data from `url` and store the result in `filename`."""
    if not os.path.exists(filename):
        print("Downloading MNIST dataset")
        urlretrieve(url, filename)

    with gzip.open(filename, 'rb') as f:
        return pickle_load(f, encoding='latin-1')


def load_data():
    """Get data with labels, split into training, validation and test set."""
    data = _load_data()
    print data
    X_train, y_train = data[0]
    X_valid, y_valid = data[1]
    X_test, y_test = data[2]
    y_train = numpy.asarray(y_train, dtype=numpy.int32)
    y_valid = numpy.asarray(y_valid, dtype=numpy.int32)
    y_test = numpy.asarray(y_test, dtype=numpy.int32)

    return dict(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
        num_examples_train=X_train.shape[0],
        num_examples_valid=X_valid.shape[0],
        num_examples_test=X_test.shape[0],
        input_dim=X_train.shape[1],
        output_dim=10,
    )


def nn_example(data):
    net1 = NeuralNet(
        layers=[('input', layers.InputLayer),
                ('hidden', layers.DenseLayer),
                ('output', layers.DenseLayer),
                ],
        # layer parameters:
        input_shape=(None, 28*28),
        hidden_num_units=100,  # number of units in 'hidden' layer
        output_nonlinearity=lasagne.nonlinearities.softmax,
        output_num_units=10,  # 10 target values for the digits 0, 1, 2, ..., 9

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,

        max_epochs=10,
        verbose=1,
        )

    # Train the network
    net1.fit(data['X_train'], data['y_train'])

    # Try the network on new data
    print("Feature vector (100-110): %s" % data['X_test'][0][100:110])
    print("Label: %s" % str(data['y_test'][0]))
    print("Predicted: %s" % str(net1.predict([data['X_test'][0]])))




# In[9]:


data = load_data()
print("Got %i testing datasets." % len(data['X_train']))
nn_example(data)



# In[11]:

data['X_train'].shape


# In[12]:

data['y_train'].shape


# In[13]:

data['y_train'][0]


# In[15]:

data['y_train'][100
               ]


# In[ ]:



