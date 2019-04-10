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

from os.path import join
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from crypr.transformers import MovingAverageTransformer, PercentChangeTransformer, PassthroughTransformer
from crypr.build import data_to_supervised
from crypr.util import get_project_path

coin = 'BTC'
data_path = join(get_project_path(), 'data', 'raw', coin + '.csv')


# In[2]:


data = pd.read_csv(data_path, index_col=0)
p(data.shape)
data.head()


# In[3]:


preprocessing_config = {
    'passthrough': ['close', 'low', 'high'],
    'moving_average': ['close', 'volumeto', 'volumefrom'],
    'target': 'close',
    'tx': 72,
    'ty': 1,
    'test_fraction': .05,
    'truncate': (6000),
    'truncate_keep_last': True,
}
pc = preprocessing_config


# In[4]:


transforms = [
    ('passthrough', PassthroughTransformer(), pc['passthrough']),
    ('ma03', MovingAverageTransformer(3), pc['moving_average']),
    ('ma06', MovingAverageTransformer(6), pc['moving_average']),
    ('ma12', MovingAverageTransformer(12), pc['moving_average']),
    ('ma24', MovingAverageTransformer(24), pc['moving_average']),
    ('ma48', MovingAverageTransformer(48), pc['moving_average']),
    ('make_target', PercentChangeTransformer(), [pc['target']]),
]
ct = ColumnTransformer(transforms, remainder='drop', n_jobs=-1)
ct = ct.fit(data)
features = ct.get_feature_names()
features


# In[5]:


arr = ct.transform(data)
arr = arr[~np.isnan(arr).any(axis=1)]
arr.view()


# In[6]:


plt.figure(); plt.plot(arr[:, features.index('passthrough__close')]); plt.title('close')
plt.figure(); plt.plot(arr[:, features.index('make_target__close')]); plt.title('target')
# plt.figure(); plt.plot(df.filter(regex='v(t|f)')); plt.title('v(t|f)')
plt.show()


# In[7]:


num_features = arr.shape[1] - pc['ty']
p('Number of Unique Features:', num_features)
p('Number of Hours per Sample:', pc['tx'])
p('Total Features per Sample:', pc['tx']*num_features)


# In[8]:


X, y = data_to_supervised(input_df=arr, target_ix=-1, Tx=pc['tx'], Ty=pc['ty'])
p(X.head(2))
p(y.head(5))


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=pc['test_fraction'], shuffle=False)
p('Train shape: ', X_train.shape)
p('Test shape: ', X_test.shape)


# In[10]:


fig, ax = plt.subplots(1, figsize=(10, 5))
ax.plot(y_train, label='train')
ax.plot(y_test, label='test')
plt.title('Target Variable: Train Test Split')
fig.legend()
plt.show()

