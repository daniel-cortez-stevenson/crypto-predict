#!/usr/bin/env python
# coding: utf-8

# # Statistical Forecasting of Closing Price
# - Work in Progress
# - AR/ARMA/ARIMA modelling
# - Based off of several good online articles

# In[32]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
p = print

import os
import datetime

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
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


# In[98]:


"""
Import Data.
"""
SYM = 'BTC'
data_path = os.path.join(get_project_path(), 'data', 'raw', SYM + '.csv')
data = pd.read_csv(os.path.join(data_path), index_col=-1)
data.head()


# In[99]:


# # log returns
# lrets = np.log(df.close/df.close.shift(1)).dropna()
# lrets.plot()
# plt.show()

# percent change
pchange = data['close'].pct_change()
pchange.plot()
plt.show()


# In[195]:


Y = data['close'].pct_change().dropna()
Y.index = list(map(lambda ix: datetime.datetime.strptime(ix, '%Y-%m-%d %H:%M:%S').timestamp(), Y.index))
max_lag = 30
freq = 'H'
forecast_steps = 21
Y.head()


# In[196]:


def tsplot(y, lags=None, figsize=(10, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        #mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        
        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return 


# In[197]:


_ = tsplot(Y, lags=max_lag)


# In[272]:


# Select best lag order for BTC returns using Autoregressive Model
ar_est_order = smt.AR(Y).select_order(maxlag=max_lag, ic='aic', trend='nc')
p('best estimated lag order = {}'.format(ar_est_order))


# In[158]:


# Fit MA(3) to BTC returns
arma_mdl = smt.ARMA(Y, order=(0, 3)).fit(maxlag=max_lag, method='mle', trend='nc', freq=freq)
p(arma_mdl.summary())


# # ARMA

# In[199]:


# Fit ARMA model to BTC returns
def _get_best_arma(TS):
    best_aic = np.inf 
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
    p('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
    return best_aic, best_order, best_mdl

arma_aic, arma_order, arma_mdl = _get_best_arma(Y)


# # ARIMA

# In[200]:


def _get_best_arima(TS):
    best_aic = np.inf 
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
    p('aic: {:6.5f} | order: {}'.format(best_aic, best_order))                    
    return best_aic, best_order, best_mdl


# In[201]:


arima_aic, arima_order, arima_mdl = _get_best_arima(Y)


# # Predict Volatility with ARIMA

# In[202]:


# Create a XXX hour forecast of BTC returns with 95%, 99% CI
f, err95, ci95 = arima_mdl.forecast(steps=forecast_steps) # 95% CI
_, err99, ci99 = arima_mdl.forecast(steps=forecast_steps, alpha=0.01) # 99% CI

idx = pd.date_range(Y.index[-1], periods=forecast_steps, freq=freq)
fc_95 = pd.DataFrame(np.column_stack([f, ci95]), 
                     index=idx, columns=['forecast', 'lower_ci_95', 'upper_ci_95'])
fc_99 = pd.DataFrame(np.column_stack([ci99]), 
                     index=idx, columns=['lower_ci_99', 'upper_ci_99'])
fc_all = fc_95.combine_first(fc_99)
fc_all.head()


# In[271]:


# Plot XXX hour forecast for crypto returns
plt.style.use('bmh')
fig = plt.figure(figsize=(9, 7))
ax = plt.gca()
styles = ['b-', '0.2', '0.75', '0.2', '0.75']

fc_all.plot(style=styles, ax=ax)
plt.fill_between(fc_all.lower_ci_95, fc_all.upper_ci_95, color='gray', alpha=0.7)
plt.fill_between(fc_all.lower_ci_99, fc_all.upper_ci_99, color='gray', alpha=0.2)
plt.title('{} Hour {} Return Forecast\nARIMA {}'.format(forecast_steps, arima_order, SYM))
plt.legend(loc='best', fontsize=10)
plt.show()


# # GARCH based on ARIMA

# In[204]:


# Now we can fit the arch model using the best fit arima model parameters

p_ = arima_order[0]
o_ = arima_order[1]
q_ = arima_order[2]

# Using student T distribution usually provides better fit
arch_mdl = arch.arch_model(arima_mdl.resid, p=p_, o=o_, q=q_, vol='GARCH', dist='StudentsT')
arch_res = arch_mdl.fit(update_freq=5, disp='off')
p(arch_res.summary())


# In[205]:


idx = pd.date_range(Y.index[-1], periods=forecast_steps, freq=freq)


# In[206]:


# Create a 21 hour forecast of BTC returns
arch_f = arch_res.forecast(horizon=forecast_steps, start=Y.index[-1], method='simulation', simulations=1000) # 95% CI


# In[207]:



sims = arch_f.simulations
fig, ax = plt.subplots()
lines = plt.plot(idx, sims.values[-1, :, :].T, color='blue', alpha=0.1)
lines[0].set_label('Simulated paths')
fig.autofmt_xdate()
plt.show()

p(np.percentile(sims.values[-1, :, -1].T, 5))
plt.hist(sims.values[-1, :, -1], bins=50)
plt.title('Distribution of Returns')
plt.show()


# In[208]:


fig = arch_res.hedgehog_plot(horizon=forecast_steps)


# In[209]:


arch_res.plot(annualize='D', scale=365)


# # From Medium
# https://medium.com/auquan/time-series-analysis-for-finance-arch-garch-models-822f87f1d755

# In[210]:


tsplot(arch_res.resid**2, lags=30)


# In[259]:


Y_ = Y.iloc[-1000:]
window = 300
foreLength = len(Y_) - window
signal = Y_[-foreLength:]
signal.head()


# In[260]:


def backtest(Y, foreLength, window, signal, order=''):
    for d in range(foreLength):

        # create a rolling window by selecting 
        # values between d+1 and d+T of S&P500 returns

        TS = Y[(1+d):(window+d)] 
        
        if order:
            best_order=order
            best_mdl=smt.ARIMA(TS, order=(best_order[0], best_order[1], best_order[2]), freq=freq).fit(
                        method='mle', trend='nc'
                    )
        else:
            # Find the best ARIMA fit 
            # set d = 0 since we've already taken log return of the series
            _, best_order, best_mdl = _get_best_arima(TS)
        
        
        #now that we have our ARIMA fit, we feed this to GARCH model
        p_ = best_order[0]
        o_ = best_order[1]
        q_ = best_order[2]
        
        if p_ == o_ == 0:
            p_ = 1
        
        am = arch.arch_model(best_mdl.resid, p=p_, o=o_, q=q_, dist='StudentsT')
        res = am.fit(update_freq=5, disp='off')

        # Generate a forecast of next day return using our fitted model
        out = res.forecast(horizon=1, start=None, align='origin')

        #Set trading signal equal to the sign of forecasted return
        # Buy if we expect positive returns, sell if negative

        signal.iloc[d] = np.sign(out.mean['h.1'].iloc[-1])
    return signal


# In[261]:


signal = backtest(Y_, foreLength, window, signal)


# In[262]:


returns = pd.DataFrame(index=signal.index, 
                       columns=['Buy and Hold', 'Strategy'])
returns['Buy and Hold'] = Y_.iloc[-foreLength:]
returns['Strategy'] = signal*returns['Buy and Hold']

eqCurves = pd.DataFrame(index=signal.index, 
                        columns=['Buy and Hold', 'Strategy'])
eqCurves['Buy and Hold'] = returns['Buy and Hold'].cumsum() + 1
eqCurves['Strategy'] = returns['Strategy'].cumsum() + 1

eqCurves['Strategy'].plot(figsize=(10, 8))
eqCurves['Buy and Hold'].plot()
plt.legend()
plt.show()


# # From Arch website

# In[273]:


from arch.univariate import ARX
ar = ARX(Y, lags=30)
print(ar.fit().summary())


# In[270]:


from arch.univariate import ARCH, GARCH
ar.volatility = GARCH(p=3, o=0, q=3)
res = ar.fit(update_freq=0, disp='off')
p(res.summary())


# In[265]:


from arch.univariate import StudentsT
ar.distribution = StudentsT()
res = ar.fit(update_freq=0, disp='off')
p(res.summary())


# In[266]:


arf = ar.forecast(horizon=forecast_steps, start=Y.index[-1], 
                  params=res.params, method='simulation')


# In[267]:


plt.plot(idx, arf.simulations.values[-1].T, 
         color='blue', alpha=0.1)
plt.show()


# In[269]:


plt.style.use('bmh')
fig, ax = plt.subplots()
plt.plot(idx, arf.simulations.variances[-1,::].T, 
         color='blue', alpha=0.1, label='simulated var')
plt.plot(idx, arf.variance.iloc[-1], 
         color='red', label='expected var')
fig.autofmt_xdate()
plt.show()


# In[225]:


# arp = ar.simulate(params=res.params, nobs=res.nobs)
# arp[['data', 'errors']].plot(alpha=0.3)

