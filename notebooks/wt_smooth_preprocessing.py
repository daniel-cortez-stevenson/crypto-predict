#!/usr/bin/env python
# coding: utf-8

# # Smoothing with Wave Transform Preprocessing

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
p = print

from os.path import join
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
from crypr.build import make_features, data_to_supervised, dwt_smoother


# In[2]:


SYM = 'BTC'
TARGET = 'close'
Tx = 72
Ty = 1
TEST_SIZE = 0.05
data_path = join(get_project_path(), 'data', 'raw', SYM + '.csv')


# In[3]:


data = pd.read_csv(data_path, index_col=0)
data.head()


# In[5]:


"""
Get percent change feature and target data.
"""
df = make_features(input_df=data, target_col='close')
X, y = data_to_supervised(input_df=df[['target__close']], target_ix=-1, Tx=Tx, Ty=Ty)
p(X.shape, y.shape)
X.head()


# In[6]:


"""
Confirm data reshape and target/feature creation was done correctly.
"""
y_values_except_last = np.squeeze(y.iloc[:-1].values)
t_minus_1_x_values_except_first = X.iloc[1:,-1].values

y_values_except_last.all() == t_minus_1_x_values_except_first.all()


# In[7]:


"""
For comparing different transformations
"""
sample_ix = 1000
sample = X.iloc[sample_ix].values


# In[8]:


"""
DWT Haar Transform
"""
smoothed_sample = dwt_smoother(sample, 'haar', smooth_factor=.4)
plt.plot(sample, label='raw')
plt.plot(smoothed_sample, label='smoothed')
plt.title('DWT Haar Smoothing')
plt.legend()
plt.show()


# In[9]:


"""
Apply the wavelet transformation smoothing to the feature data.
"""
wt_type = 'haar'
smoothing = .4
X_smooth = np.apply_along_axis(func1d=lambda x: dwt_smoother(x, wt_type, smooth_factor=smoothing), 
                               axis=-1, arr=X)

assert X_smooth.shape == X.shape


# In[10]:


"""
Train Test Split.
"""
X_train, X_test, y_train, y_test = train_test_split(X_smooth, y, test_size=TEST_SIZE, shuffle=False)


# In[11]:


# """
# Save data.
# """
# np.save(arr=X_train, file='../data/processed/X_train_{}_{}_smooth_{}'.format(SYM, wt_type, Tx))
# np.save(arr=X_test,  file='../data/processed/X_test_{}_{}_smooth_{}'.format(SYM, wt_type, Tx))
# np.save(arr=y_train, file='../data/processed/y_train_{}_{}_smooth_{}'.format(SYM, wt_type, Tx))
# np.save(arr=y_test,  file='../data/processed/y_test_{}_{}_smooth_{}'.format(SYM, wt_type, Tx))

