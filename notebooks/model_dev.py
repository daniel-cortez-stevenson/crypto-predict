#!/usr/bin/env python
# coding: utf-8

# # Model Development

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
p = print

from os.path import join
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, MultiTaskLassoCV, Ridge, RidgeCV, MultiTaskElasticNet, MultiTaskElasticNetCV, ElasticNet, ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

from crypr.util import get_project_path


# In[2]:


"""
Import Data.
"""
SYM = 'BTC'
Ty = 1
Tx = 72
MAX_LAG = 72
wavelet = 'haar_smooth'
data_dir = join(get_project_path(), 'data', 'processed')
models_dir = join(get_project_path(), 'models')

X_train = np.load(join(data_dir, 'X_train_{}.npy'.format(SYM)))
Y_train = np.load(join(data_dir, 'y_train_{}.npy'.format(SYM)))
X_test = np.load(join(data_dir, 'X_test_{}.npy'.format(SYM)))
Y_test = np.load(join(data_dir, 'y_test_{}.npy'.format(SYM)))

N_FEATURES = X_train.shape[2]

p(X_train.shape)
X_train = X_train.reshape((-1, Tx*N_FEATURES))
X_test = X_test.reshape((-1, Tx*N_FEATURES))
p(X_train.shape)


# # Let's try a simple linear regression model

# In[3]:


lr_model = LinearRegression()
lr_model = lr_model.fit(X_train, Y_train)
lr_predict = lr_model.predict(X_test)

p(mean_absolute_error(y_pred=lr_predict, y_true=Y_test))
p(mean_squared_error(y_pred=lr_predict, y_true=Y_test))


# In[4]:


# Save model
# with open(join(models_dir, 'linear_model_{}.pkl'.format(SYM)), 'wb') as output_file:
#     s = pickle.dump(lr_model, output_file)


# # Other Linear Regression

# ## Lasso

# In[5]:


lasso_params = {
    'alpha': [0.01],
}

lasso_model = Lasso(alpha=0.01, max_iter=10000)
lasso_model = lasso_model.fit(X=X_train, y=Y_train)

lasso_predict = lasso_model.predict(X_test)

p(mean_absolute_error(lasso_predict, Y_test))
p(mean_squared_error(lasso_predict, Y_test))


# ## Ridge
# 

# In[6]:


ridge_model = Ridge(alpha=0.01)
ridge_model = ridge_model.fit(X=X_train, y=Y_train)

ridge_predict = ridge_model.predict(X_test)

p(mean_absolute_error(ridge_predict, Y_test))
p(mean_squared_error(ridge_predict, Y_test))


# ## Elastic Net

# In[7]:


enet_params = {
    'alpha': [1e-7],
}

enet_model = MultiTaskElasticNetCV(alphas=enet_params['alpha'])
enet_model = enet_model.fit(X=X_train, y=Y_train)

enet_predict = enet_model.predict(X_test)

p(mean_absolute_error(enet_predict, Y_test))
p(mean_squared_error(enet_predict, Y_test))

