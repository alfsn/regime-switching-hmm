#!/usr/bin/env python
# coding: utf-8

# ## Startup

# In[71]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import norm
import scipy.stats as scs


import os
import pickle


# In[2]:


np.random.seed(42)


# In[3]:


from scripts.params import get_params

params = get_params()


# In[18]:


dataroute = os.path.join("..", "data")
processedroute = os.path.join("...", "processed")
resultsroute = os.path.join("..", "results")
descriptivegraphsroute=os.path.join(resultsroute, "graphs", "descriptive")


# ## Data Retrieval

# In[5]:


name = f'finaldf_train_{params["tablename"]}.pickle'
filename = os.path.join(dataroute, name)
with open(filename, "rb") as handle:
    df = pickle.load(handle)


# In[6]:


df.head()


# ## Descriptive graphs

# In[24]:


log_rets_list=[]
vol_list=[]
for column in df.columns: 
    if column.endswith("log_rets"):
        log_rets_list.append(column)
    if column.endswith("gk_vol"):
        vol_list.append(column)


# In[25]:


sb.set_style(style='darkgrid')
sb.set_palette(sb.color_palette(palette='deep'))


# In[26]:


for column in log_rets_list:
    fig=df[column].plot(title=column).get_figure()
    fig.savefig(os.path.join(descriptivegraphsroute, "log_rets", f"{column}.png"))
    plt.close()


# In[27]:


for column in vol_list:
    fig=df[column].plot(title=column).get_figure()
    fig.savefig(os.path.join(descriptivegraphsroute, "gk_vol", f"{column}.png"))
    plt.close()


# ### Autocorrelograms

# In[38]:


acf_lags=252


# In[39]:


def save_acf(column, path):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    plot_acf(df[column],
                lags=acf_lags,                 # Cuantos lags busco autocorrelacionar
                zero=False,                # Si tomo el lag cero
                alpha=0.05,                # El rango de certeza marcado en azul
                use_vlines=False,          # Lineas verticales que conectan cada punto con el eje x
                ax=ax1)                    # La posicion en la figura
    ax1.grid(True)
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('Autocorrelation')
    ax1.set_title(f'Autocorrelation of {column}')
    fig.savefig(path)
    plt.close()


# In[40]:


for column in log_rets_list:
    save_acf(column, 
             os.path.join(
                 descriptivegraphsroute, 
                 "acf", "log_rets", f"{column}.png"))


# In[41]:


for column in vol_list:
    save_acf(column, 
             os.path.join(
                 descriptivegraphsroute, 
                 "acf", "gk_vol", f"{column}.png"))


# In[45]:


def save_pacf(column, path):
    """
    """
    fig = plt.figure(figsize=[13.6, 10.2])
    ax1 = fig.add_subplot(1, 1, 1)
    plot_pacf(df[column],
                method='ywa',                 # Metodo de Yule Walker con correccion de sesgo por autocovarianzas
                lags=acf_lags,                # Cuantos lags busco autocorrelacionar
                zero=False,                   # Si tomo el lag cero
                alpha=0.05,                   # El rango de certeza marcado en azul
                use_vlines=False,             # Lineas verticales que conectan cada punto con el eje x
                ax=ax1)                       # La posicion en la figura
    ax1.grid(True)
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('Partial Autocorrelation')
    ax1.set_title(f'Partial Autocorrelation of {column}')

    fig.savefig(path)
    plt.close()


# In[46]:


for column in log_rets_list:
    save_pacf(column, 
             os.path.join(
                 descriptivegraphsroute, 
                 "pacf", "log_rets", f"{column}.png"))


# In[48]:


for column in vol_list:
    save_acf(column, 
             os.path.join(
                 descriptivegraphsroute, 
                 "pacf", "gk_vol", f"{column}.png"))


# In[66]:


def save_hist_normal(column, path):
    fig = plt.figure(figsize=[13.6, 5.1])
    ax1 = fig.add_subplot(1, 1, 1)

    sb.distplot(df[column].fillna(0),
                ax=ax1,
                hist=True,
                bins=int(np.ceil(np.log2(len(df[column].index)) + 15)),
                label='Observed data KDE',
                fit=norm,
                fit_kws = {"color":"r", "lw":3, "label":"Fitted Normal"})
    plt.legend()
    plt.grid(True)
    plt.xlabel('Log Returns')
    plt.ylabel('Frequency')
    plt.title(f'Histogram for return frequency for {column}')
    fig.savefig(path)
    plt.close()


# In[67]:


for column in log_rets_list:
    save_hist_normal(
        column,
        os.path.join(
            descriptivegraphsroute, 
            "histograms", f"{column}.png"))


# In[68]:


for column in vol_list:
    save_hist_normal(
        column,
        os.path.join(
            descriptivegraphsroute, 
            "histograms", f"{column}.png"))


# In[73]:


def analyze_skew_kurt(dataframe):
    results = pd.DataFrame(index=dataframe.columns, 
                           columns=['Skewness', 'Skewness P-Value', 'Kurtosis', 'Kurtosis P-Value', 'Normaltest Stat', 'Normaltest P-Value'])
    for column in dataframe.columns:
        skew_val, skew_pval = scs.skewtest(dataframe[column], nan_policy='omit')
        kurt_val, kurt_pval = scs.kurtosistest(dataframe[column], nan_policy='omit')
        normtest_stat, normtest_pval = scs.normaltest(dataframe[column], nan_policy='omit')
        results.loc[column] = [skew_val, skew_pval, kurt_val, kurt_pval, normtest_stat, normtest_pval]

    return results


# In[74]:


analyze_skew_kurt(df[log_rets_list].fillna(0))
