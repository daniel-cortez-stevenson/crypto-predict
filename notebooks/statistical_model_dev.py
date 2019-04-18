#!/usr/bin/env python
# coding: utf-8

# # Statistical Forecasting of Close Value, Returns, and Volatility
# - Work in Progress
# - AR/ARMA/ARIMA/GARCH modelling

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
p = print

from os.path import join
from datetime import datetime

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from crypr.util import get_project_path

import statsmodels
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm

import scipy.stats as scs

import arch
from arch.univariate import ARX, GARCH
from arch.univariate import StudentsT


# In[2]:


"""
Set plotting style.
"""
plt.style.use('bmh')


# In[3]:


"""
Import Data.
"""
SYM = 'BTC'
data_path = join(get_project_path(), 'data', 'raw', SYM + '.csv')
data = pd.read_csv(join(data_path), usecols=['close', 'time'], index_col='time')
data.head()


# In[4]:


"""
Calculate signals.
"""
data['log_returns'] = np.log(data.close / data.close.shift(1))
data['percent_change'] = data.close.pct_change()
data.head()


# In[5]:


"""
Set analysis vars.
"""
Y = data['log_returns'].dropna()
Y.index = pd.DatetimeIndex(pd.date_range(Y.index[0]*1000000000, freq='H', periods=len(Y.index)))

max_lag = 30
freq = 'H'
forecast_steps = 21
forecast_idx = pd.date_range(Y.index[-1], periods=forecast_steps, freq=freq)

Y.head()


# In[6]:


def tsplot(y, lags=None, figsize=(10, 8), style='bmh') -> None: 
    fig = plt.figure(figsize=figsize)
    layout = (3, 2)

    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    ts_ax.set_title('Time Series Analysis Plots')
    y.plot(ax=ts_ax)

    acf_ax = plt.subplot2grid(layout, (1, 0))
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)

    pacf_ax = plt.subplot2grid(layout, (1, 1))
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)

    qq_ax = plt.subplot2grid(layout, (2, 0))
    qq_ax.set_title('QQ Plot') 
    sm.qqplot(y, line='s', ax=qq_ax)

    pp_ax = plt.subplot2grid(layout, (2, 1))
    scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

    plt.tight_layout()

tsplot(Y, lags=max_lag)


# In[7]:


# Select best lag order for crypto returns using Autoregressive (AR) Model
ar_lag_order_estimate = smt.AR(Y, freq=freq).select_order(maxlag=max_lag, ic='aic', trend='nc')
p('best estimated lag order =', ar_lag_order_estimate)


# In[8]:


# Fit MA(3) to crypto returns
arma_mdl = smt.ARMA(Y, order=(0, 3), freq=freq).fit(maxlag=max_lag, method='mle', trend='nc')
arma_mdl.summary()


# # ARMA

# In[9]:


def _get_best_arma(TS):
    best_aic = float('inf')
    best_order = None
    best_mdl = None

    for i in range(5):
        for j in range(5):
            try:
                tmp_mdl = smt.ARMA(TS, order=(i, j)).fit(method='mle', trend='nc', freq=freq)
                tmp_aic = tmp_mdl.aic
                if tmp_aic < best_aic:
                    best_aic = tmp_aic
                    best_order = (i, j)
                    best_mdl = tmp_mdl
            except: continue
    p('ARMA Results:', 'best aic = {:6.5f} | best order = {}'.format(best_aic, best_order))
    return best_aic, best_order, best_mdl

arma_aic, arma_order, arma_mdl = _get_best_arma(Y)


# # ARIMA

# In[10]:


def _get_best_arima(TS):
    best_aic = float('inf')
    best_order = None
    best_mdl = None

    pq_rng = range(5)
    d_rng = range(2)
    for i in pq_rng:
        for d in d_rng:
            for j in pq_rng:
                try:
                    tmp_mdl = smt.ARIMA(TS, order=(i, d, j)).fit(method='mle', trend='nc', freq=freq)
                    tmp_aic = tmp_mdl.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (i, d, j)
                        best_mdl = tmp_mdl
                except: continue
    p('ARIMA Results: best aic = {:6.5f} | best order = {}'.format(best_aic, best_order))                    
    return best_aic, best_order, best_mdl


# In[11]:


arima_aic, arima_order, arima_mdl = _get_best_arima(Y)


# # Predict Volatility with ARIMA

# In[12]:


# Create an hourly forecast of crypto returns with 95%, 99% CI
f, err95, ci95 = arima_mdl.forecast(steps=forecast_steps) # 95% CI
_, err99, ci99 = arima_mdl.forecast(steps=forecast_steps, alpha=0.01) # 99% CI

fc_95 = pd.DataFrame(np.column_stack([f, ci95]), 
                     index=forecast_idx, columns=['forecast', 'lower_ci_95', 'upper_ci_95'])
fc_99 = pd.DataFrame(np.column_stack([ci99]), 
                     index=forecast_idx, columns=['lower_ci_99', 'upper_ci_99'])
fc_all = fc_95.combine_first(fc_99)
fc_all.head()


# In[13]:


# Plot forecast for crypto returns
fig, ax = plt.subplots(figsize=(9, 7))
styles = ['b-', '0.2', '0.75', '0.2', '0.75']

fc_all.plot(style=styles, ax=ax)

plt.fill_between(fc_all.lower_ci_95, fc_all.upper_ci_95, color='gray', alpha=0.7)
plt.fill_between(fc_all.lower_ci_99, fc_all.upper_ci_99, color='gray', alpha=0.2)

plt.title('{} Hour {} Returns Forecast\nARIMA {}'.format(forecast_steps, SYM, arima_order))
plt.legend(loc='best', fontsize=10)
plt.show()


# # GARCH based on ARIMA

# In[14]:


# fit the arch model using the best fit arima model parameters
p_, o_, q_ = arima_order

# Using student T distribution usually provides better fit
arch_mdl = arch.arch_model(arima_mdl.resid, p=p_, o=o_, q=q_, vol='GARCH', dist='StudentsT')
arch_res = arch_mdl.fit(update_freq=5, disp='off')
arch_res.summary()


# In[15]:


# Create a 21 hour forecast of crypto returns 
arch_f = arch_res.forecast(horizon=forecast_steps, start=Y.index[-1], method='simulation', simulations=1000) 


# In[16]:


arch_f_sims = arch_f.simulations

p('Percentile of simulation values = {:.6}'.format(np.percentile(arch_f_sims.values[-1, :, -1].T, 5)))

fig, ax = plt.subplots(2, sharex=False, figsize=(8, 5))

plt.sca(ax[0])
plt.plot(forecast_idx, arch_f_sims.values[-1, :, :].T, color='blue', alpha=0.1)
plt.title('Simulated paths')

plt.sca(ax[1])
plt.hist(arch_f_sims.values[-1, :, -1], bins=50)
plt.title('Distribution of Returns')

plt.tight_layout()


# In[17]:


fig = arch_res.hedgehog_plot(horizon=forecast_steps)
fig.autofmt_xdate()


# In[18]:


fig = arch_res.plot(annualize=freq, scale=365)
fig.autofmt_xdate()


# # From Medium
# https://medium.com/auquan/time-series-analysis-for-finance-arch-garch-models-822f87f1d755

# In[19]:


tsplot(arch_res.resid**2, lags=max_lag)


# In[20]:


Y_ = Y.iloc[-1000:]
window = 72
foreLength = forecast_steps + window
signal = Y_[-foreLength:]
signal.head()


# In[21]:


def backtest(Y, foreLength, window, signal, frequency, order=None):
    for d in range(foreLength):
        TS = Y[(d + 1):(d + window)] 
        
        if order:
            best_mdl = smt.ARIMA(TS, order=order, freq=frequency).fit(method='mle', trend='nc')
        else:
            _, order, best_mdl = _get_best_arima(TS)
        
        p_, o_, q_ = order
        if p_ == o_ == 0:
            p_ = 1
        
        arch_mdl = arch.arch_model(best_mdl.resid, p=p_, o=o_, q=q_, dist='StudentsT')
        res = arch_mdl.fit(update_freq=5, disp='off')
        forecast = res.forecast(horizon=1, start=None, align='origin')
        # Set trading signal equal to the sign of forecasted return
        signal.iloc[d] = np.sign(forecast.mean['h.1'].iloc[-1])
    return signal


# In[22]:


signal = backtest(Y_, foreLength, window, signal, freq)


# In[23]:


returns = pd.DataFrame(index=signal.index, columns=['Buy and Hold', 'Strategy'])
returns['Buy and Hold'] = Y_.iloc[-foreLength:]
returns['Strategy'] = signal * returns['Buy and Hold']

eqCurves = pd.DataFrame(index=signal.index, columns=['Buy and Hold', 'Strategy'])
eqCurves['Buy and Hold'] = returns['Buy and Hold'].cumsum() + 1
eqCurves['Strategy'] = returns['Strategy'].cumsum() + 1

fig, ax = plt.subplots(figsize=(10, 8))
eqCurves['Strategy'].plot()
eqCurves['Buy and Hold'].plot()
plt.legend()
plt.show()


# # From Arch website

# In[24]:


ar = ARX(Y, lags=max_lag)
ar.fit().summary()


# In[25]:


ar.volatility = GARCH(p=3, o=0, q=3)
res = ar.fit(update_freq=0, disp='off')
res.summary()


# In[26]:


ar.distribution = StudentsT()
res = ar.fit(update_freq=0, disp='off')
res.summary()


# In[27]:


ar_forecast = ar.forecast(horizon=forecast_steps, start=Y.index[-1], params=res.params, method='simulation')


# In[28]:


ar_forecast_simulations = ar_forecast.simulations.values[-1].T
plt.plot(forecast_idx, ar_forecast_simulations, color='blue', alpha=0.1)
plt.show()


# In[29]:


fig, ax = plt.subplots(figsize=(10, 8))
plt.plot(forecast_idx, ar_forecast.simulations.variances[-1,::].T, color='blue', alpha=0.1, label='simulated var')
plt.plot(forecast_idx, ar_forecast.variance.iloc[-1], color='red', label='expected var')
fig.autofmt_xdate()
plt.show()

