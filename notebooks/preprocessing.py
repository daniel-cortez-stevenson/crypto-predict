#!/usr/bin/env python
# coding: utf-8

# # Example Preprocessing Flow
# - Calculate the target as % change between sucessive raw 'close' values
# - Calculate moving averages (feature engineering)
# - Represent the input X as a time series (data_to_supervised function)
# - Train-test split and visualize

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
p = print

import os
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split

from crypr.build import truncate, calc_target, calc_volume_ma, data_to_supervised, save_preprocessing_output
from crypr.util import get_project_path

coin = 'BTC'
data_path = os.path.join(get_project_path(), 'data', 'raw', coin + '.csv')


# In[2]:


data = pd.read_csv(data_path, index_col=0)
p(data.shape)
data.head()


# In[3]:


MOVING_AVERAGE_LAGS = [6, 12, 24, 48, 72]
TARGET = 'close'
Tx = 72
Ty = 1
TEST_SIZE = 0.05


# In[4]:


df = data     .drop(['time', 'timestamp'], axis=1)     .pipe(calc_target, TARGET)    .pipe(calc_volume_ma, MOVING_AVERAGE_LAGS)    .dropna(how='any', axis=0)
    
plt.figure(); plt.plot(df.close)
plt.figure(); plt.plot(df.target)
plt.figure(); plt.plot(df.filter(regex='v(t|f)'))
df.head()


# In[5]:


N_FEATURES = len(df.columns)
N_FEATURES


# In[6]:


X, y = data_to_supervised(input_df=df, Tx=Tx, Ty=Ty)
X.head()


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)
p('Train shape: ', X_train.shape)
p('Test shape: ', X_test.shape)


# In[8]:


fig, ax = plt.subplots(2, figsize=(12, 8))

ax[0].plot(y_train, label='train')
ax[0].plot(y_test, label='test')

ax[1].plot(X_train.iloc[:,-N_FEATURES:])
ax[1].plot(X_test.iloc[:,-N_FEATURES:])
fig.legend()
plt.show()

