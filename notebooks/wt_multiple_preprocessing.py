#!/usr/bin/env python
# coding: utf-8

# # Multiple Variable Wavelet Preprocessing

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
p = print

from os.path import join

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from scipy import signal
import pywt

from crypr.util import get_project_path
from crypr.build import make_features, data_to_supervised, make_3d
from crypr.transformers import PassthroughTransformer, HaarSmoothTransformer


# In[2]:


SYM = 'BTC'
TARGET = 'close'
Tx = 72
Ty = 1
TEST_SIZE = 0.05

data_path = join(get_project_path(), 'data', 'raw', SYM + '.csv')
data = pd.read_csv(data_path, index_col=0)

"""
Train Test Split.
"""
data_train, data_test = train_test_split(data, test_size=TEST_SIZE, shuffle=False)
data_train = data_train.dropna()
data_test = data_test.dropna().iloc[:-1]

p(data_train.shape, data_test.shape)
data_test.head()


# In[3]:


"""
Get features.
"""
feature_data_train = make_features(input_df=data_train, target_col='close', ma_cols=['volumeto', 'volumefrom'], ma_lags=[3, 6, 12])
feature_data_test = make_features(input_df=data_test, target_col='close', ma_cols=['volumeto', 'volumefrom'], ma_lags=[3, 6, 12])

feature_data_train.dropna(inplace=True)
feature_data_test.dropna(inplace=True)

feature_data_test.head()


# In[4]:


"""
Apply DWT Smooth.
"""
transformers = [
    ('haar_smooth', HaarSmoothTransformer(.05), list(feature_data_train.columns)),
    ('orig', PassthroughTransformer(), ['target__close']),
]

ct = ColumnTransformer(transformers=transformers, n_jobs=-1)

smooth_arr_train = ct.fit_transform(feature_data_train)
smooth_data_train = pd.DataFrame(smooth_arr_train, columns=ct.get_feature_names())

smooth_arr_test = ct.fit_transform(feature_data_test)
smooth_data_test = pd.DataFrame(smooth_arr_test, columns=ct.get_feature_names())

smooth_data_train.plot(); plt.show()


# In[5]:


"""
Make time-series data.
"""
X_train, y_train = data_to_supervised(input_df=smooth_data_train, target_ix=-1, Tx=Tx, Ty=Ty)
X_test, y_test = data_to_supervised(input_df=smooth_data_test, target_ix=-1, Tx=Tx, Ty=Ty)
p(X_train.head())
p(y_train.head())

"""
Reshape the data into 3d array.
"""
X_train = make_3d(X_train, tx=Tx, num_channels=len(list(feature_data_train.columns)) + 1)
X_test = make_3d(X_test, tx=Tx, num_channels=len(list(feature_data_test.columns)) + 1)

X_train.view()


# In[6]:


"""
Save data.
"""
output_dir = join(get_project_path(), 'data', 'processed')

np.save(arr=X_train, file=join(output_dir, 'X_train_multiple_smooth_{}'.format(SYM)))
np.save(arr=X_test, file=join(output_dir, 'X_test_multiple_smooth_{}'.format(SYM)))
np.save(arr=y_train, file=join(output_dir, 'y_train_multiple_smooth_{}'.format(SYM)))
np.save(arr=y_test, file=join(output_dir, 'y_test_multiple_smooth_{}'.format(SYM)))

