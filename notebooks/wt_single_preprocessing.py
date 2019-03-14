#!/usr/bin/env python
# coding: utf-8

# # Wavelet Transform Coefficient Preprocessing

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
p = print

import os
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
from crypr.build import make_single_feature, data_to_supervised


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
df = make_single_feature(input_df=data, target_col='close')
p(df.head())
X, y = data_to_supervised(input_df=df[['target']], Tx=Tx, Ty=Ty)
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
Spectrogram Analysis
"""
rows = 2
cols = 12

fig, ax = plt.subplots(rows,cols, figsize=(15,5))

for r in range(rows):
    for c in range(cols):
        plt.sca(ax[r][c])
        pxx, freqs, bins, im = plt.specgram(X.iloc[(sample_ix + r*c)], 12, 1, noverlap=11)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.title((c + 1) + r*cols)
sns.despine(left=True, bottom=True)
plt.show()


# In[7]:


"""
CWT Ricker
"""
N = 14
# T = 1.0 / N
# START=1000
widths = np.arange(1, (N + 1))

rows = 2
cols = 8
fig, ax = plt.subplots(rows, cols, figsize=(12, 7))
for r in range(rows):
    for c in range(cols):
        plt.sca(ax[r][c])
        cwtmatr = signal.cwt(X.iloc[(sample_ix + r*c)], wavelet=signal.ricker, widths=widths)
        plt.imshow(cwtmatr, extent=[-1, 1, 1, (N + 1)], cmap='PRGn', aspect='auto',
            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.title((c + 1) + r*cols)
sns.despine(left=True, bottom=True)
plt.suptitle('Continuous Wavelet Transform (Ricker)')
plt.show() 


# In[8]:


"""
CWT Morlet Transform
"""

rows = 2
cols = 8

fig, ax = plt.subplots(rows,cols, figsize=(15,5))

for r in range(rows):
    for c in range(cols):
        plt.sca(ax[r][c])
        cwtmatr, freqs = pywt.cwt(X.iloc[(sample_ix + r*c)], scales=widths, wavelet='mexh')
        plt.imshow(cwtmatr, extent=[-1, 1, 1, (N + 1)], cmap='PRGn', aspect='auto',
                   vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.title((c + 1) + r*cols)
sns.despine(left=True, bottom=True)
plt.suptitle('Continuous Wavelet Transform (Morlet)')
plt.show() 


# In[9]:


"""
Apply the wave transformation to the feature data.
"""
wt_type = 'MORLET'
p('Applying {} transform ...'.format(wt_type))
N = 28
widths = np.arange(1, (N + 1))

if wt_type == 'RICKER':
    wt_transform_fun = lambda x: signal.cwt(x, wavelet=signal.ricker, widths=widths)
elif wt_type == 'MORLET':
    wt_transform_fun = lambda x: pywt.cwt(x, scales=widths, wavelet='morl')[0]
else:
    raise NotImplementedError
    
X_wt_coef = np.apply_along_axis(func1d=wt_transform_fun, axis=-1, arr=X)

p('Old shape: ', X.shape)
p('New shape: ', X_wt_coef.shape)


# In[10]:


"""
Train Test Split.
"""
X_train, X_test, y_train, y_test = train_test_split(X_wt_coef, y, test_size=TEST_SIZE, shuffle=False)


# In[11]:


"""
Save data.
"""
output_dir = os.path.join(get_project_path(), 'data', 'processed')

np.save(arr=X_train, allow_pickle=True, 
        file=os.path.join(output_dir, 'X_train_{}_{}_{}x{}'.format(SYM, wt_type, N, Tx)))
np.save(arr=X_test, allow_pickle=True,
        file=os.path.join(output_dir, 'X_test_{}_{}_{}x{}'.format(SYM, wt_type, N, Tx)))
np.save(arr=y_train, allow_pickle=True,
        file=os.path.join(output_dir, 'y_train_{}_{}_{}x{}'.format(SYM, wt_type, N, Tx)))
np.save(arr=y_test, allow_pickle=True,
        file=os.path.join(output_dir, 'y_test_{}_{}_{}x{}'.format(SYM, wt_type, N, Tx)))

