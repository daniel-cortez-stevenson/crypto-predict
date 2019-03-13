#!/usr/bin/env python
# coding: utf-8

# # What's the data like?

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
p = print
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import datetime
import seaborn as sns
import os, dotenv

coin = 'BTC'

project_path = os.path.dirname(dotenv.find_dotenv())
raw_data_path = os.path.join(project_path, 'data', 'raw', coin + '.csv')


# In[2]:


"""
load data
"""
df = pd.read_csv(raw_data_path, index_col=0)
p('data shape is: ', df.shape)
df.head()


# In[3]:


"""
descriptive stats
"""
df.describe()


# In[4]:


"""
Confirm that no NA values are present
"""
p('nan values in data: ', df.dropna().shape != df.shape)


# In[5]:


"""
Take a small sample for plotting
"""
sample = df.sample(n=1000, replace=False)


# In[6]:


"""
Check relative distribution of data features
"""
sns.boxplot(data=sample[['close', 'high', 'low', 'open', 'volumefrom']])


# In[7]:


"""
Plot data features against time. Making sure that the data set was created correctly / sanity check.
"""

pltdf = sample.copy().     melt(id_vars='time')

fig, ax = plt.subplots(3, sharex=True, figsize=(10, 14))


sns.pointplot(ax=ax[0], data=pltdf[pltdf.variable.isin(['low', 'high', 'close'])], 
              x='time', y='value', hue='variable', linestyles='-', markers='')
sns.pointplot(ax=ax[1], data=pltdf[pltdf.variable.isin(['volumefrom'])], 
              x='time', y='value', linestyles='-', markers='')
sns.pointplot(ax=ax[2], data=pltdf[pltdf.variable.isin(['volumeto'])], 
              x='time', y='value', linestyles='-', markers='')

