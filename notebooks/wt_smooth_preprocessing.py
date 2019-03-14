#!/usr/bin/env python
# coding: utf-8

# # Smoothing with Wave Transform Preprocessing

# In[6]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
p = print

import os
import gc
import pickle

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from scipy.fftpack import fft, fftfreq, fftshift
from scipy import signal
import pywt

from crypr.util import get_project_path
from crypr.build import make_single_feature, series_to_predict_matrix, make_features, data_to_supervised
from crypr.wavelets import *


# In[7]:


SYM = 'BTC'
TARGET = 'close'
Tx = 72
Ty = 1
TEST_SIZE = 0.05
data_path = os.path.join(get_project_path(), 'data', 'raw', SYM + '.csv')


# In[10]:


data = pd.read_csv(data_path, index_col=0)
data.head()


# In[11]:


"""
Get percent change feature and target data.
"""
df = make_single_feature(input_df=data, target_col='close')
X, y = data_to_supervised(input_df=df[['target']], Tx=Tx, Ty=Ty)
p(X.shape, y.shape)
X.head()


# In[12]:


"""
Confirm data reshape and target/feature creation was done correctly.
"""
y_values_except_last = np.squeeze(y.iloc[:-1].values)
t_minus_1_x_values_except_first = X.iloc[1:,-1].values

y_values_except_last.all() == t_minus_1_x_values_except_first.all()


# In[13]:


"""
For comparing different transformations
"""
sample_ix = 1000
sample = X.iloc[sample_ix].values


# In[14]:


"""
DWT Haar Transform
"""
from crypr.
def dwt_smooth(x, wavelet):
    cA, cD = pywt.dwt(x, wavelet)
    
    def make_threshold(x):
        return np.std(x)*np.sqrt(2*np.log(x.size))
    
    cAt = pywt.threshold(cA, make_threshold(cA), mode="soft")                
    cDt = pywt.threshold(cD, make_threshold(cD), mode="soft")                
    tx = pywt.idwt(cAt, cDt, wavelet)
    return tx

plt.plot(sample, label='raw')
plt.plot(dwt_smooth(sample, 'haar'), label='smoothed')
plt.title('DWT Haar Smoothing')
plt.legend()
plt.show()


# In[17]:


"""
Apply the wavelet transformation smoothing to the feature data.
"""
wt_type = 'haar'
X_smooth = np.apply_along_axis(func1d=lambda x: dwt_smooth(x, wt_type), axis=-1, arr=X)

assert X_smooth.shape == X.shape


# In[18]:


"""
Train Test Split.
"""
X_train, X_test, y_train, y_test = train_test_split(X_smooth, y, test_size=TEST_SIZE, shuffle=False)


# In[14]:


# """
# Save data.
# """
# np.save(arr=X_train, file='../data/processed/X_train_{}_{}_smooth_{}'.format(SYM, wt_type, Tx))
# np.save(arr=X_test,  file='../data/processed/X_test_{}_{}_smooth_{}'.format(SYM, wt_type, Tx))
# np.save(arr=y_train, file='../data/processed/y_train_{}_{}_smooth_{}'.format(SYM, wt_type, Tx))
# np.save(arr=y_test,  file='../data/processed/y_test_{}_{}_smooth_{}'.format(SYM, wt_type, Tx))

