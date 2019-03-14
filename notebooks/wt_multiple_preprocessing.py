#!/usr/bin/env python
# coding: utf-8

# # Multiple Variable Wavelet Preprocessing

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
p = print

import os

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from scipy import signal
import pywt

from crypr.util import get_project_path
from crypr.build import make_features, data_to_supervised


# In[2]:


SYM = 'BTC'
TARGET = 'close'
Tx = 72
Ty = 1
TEST_SIZE = 0.05

data_path = os.path.join(get_project_path(), 'data', 'raw', SYM + '.csv')
data = pd.read_csv(data_path, index_col=0)
data.head()


# In[3]:


"""
Get percent change feature and target data.
"""
df = make_features(input_df=data, target_col='close', moving_average_lags=[])
X, y = data_to_supervised(input_df=df, Tx=Tx, Ty=Ty)
p(X.shape, y.shape)
X.head()


# In[4]:


"""
Confirm data reshape and target/feature creation was done correctly.
"""
y_values_except_last = np.squeeze(y.iloc[:-1].values)
t_minus_1_x_values_except_first = X.iloc[1:,-1].values

y_values_except_last.all() == t_minus_1_x_values_except_first.all()


# In[5]:


"""
For comparing different transformations
"""
sample_ix = 1000


# In[6]:


"""
Reshape the data into 3d array if multiple variables.
"""
X = X.values.reshape((X.shape[0], -1, Tx))
p(X.shape)


# In[7]:


"""
Apply the wave transformation to the feature data.
"""
wt_type = 'DWT_HAAR'
p('Applying {} transform ...'.format(wt_type))

if wt_type == 'RICKER':
    wt_transform_fun = lambda x: signal.cwt(x, wavelet=signal.ricker, widths=widths)
elif wt_type == 'HAAR':
    wt_transform_fun = lambda x: Haar(x).getpower()
elif wt_type == 'DWT_HAAR':
    wt_transform_fun = lambda x: np.stack(pywt.dwt(x, 'haar'))
else:
    raise NotImplementedError
    
X_wt = np.apply_along_axis(func1d=wt_transform_fun, axis=-1, arr=X)

X_wt.shape


# In[8]:


"""
Condense wavelet features if multiple features analyzed.
"""
X_wt = X_wt.reshape((X_wt.shape[0], X_wt.shape[1]*X_wt.shape[2], X_wt.shape[-1]))
N = X_wt.shape[-2:]
X_wt.shape, N


# In[9]:


"""
Reshape the data so Tx is the 2nd dimension.
"""
X_wt_rs = X_wt.swapaxes(-1,-2)
p(X_wt_rs.shape)


# In[10]:


"""
Train Test Split.
"""
X_train, X_test, y_train, y_test = train_test_split(X_wt_rs, y, test_size=TEST_SIZE, shuffle=False)


# In[11]:


"""
Save data.
"""
output_dir = os.path.join(get_project_path(), 'data', 'processed')

np.save(arr=X_train, allow_pickle=True, 
        file=os.path.join(output_dir, '.X_train_{}_{}_{}x{}'.format(SYM, wt_type, Tx, N)))
np.save(arr=X_test, allow_pickle=True, 
        file=os.path.join(output_dir, 'X_test_{}_{}_{}x{}'.format(SYM, wt_type, Tx, N)))
np.save(arr=y_train, allow_pickle=True, 
        file=os.path.join(output_dir, 'y_train_{}_{}_{}x{}'.format(SYM, wt_type, Tx, N)))
np.save(arr=y_test, allow_pickle=True, 
        file=os.path.join(output_dir, 'y_test_{}_{}_{}x{}'.format(SYM, wt_type, Tx, N)))

