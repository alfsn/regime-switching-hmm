#!/usr/bin/env python
# coding: utf-8

# ## Startup

# In[73]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, coint
from statsmodels.tools.eval_measures import rmse, aic

import os
import pickle


# In[74]:


np.random.seed(42)


# In[75]:


dataroute=os.path.join("..",  "data")
resultsroute=os.path.join("..",  "results")


# ## Data Retrieval

# In[76]:


start='2013-01-01'
end="2023-06-01"

name=f'processed_dataset_{start}_{end}.pickle'
filename=os.path.join(dataroute, name)
with open(filename, 'rb') as handle:
    data=pickle.load(handle)
    
name=f'finaldf_dataset_{start}_{end}.pickle'
filename=os.path.join(dataroute, name)
with open(filename, 'rb') as handle:
    df=pickle.load(handle)


# In[77]:


df.head(3)


# ## VAR Training

# In[78]:


df.columns


# In[79]:


log_rets_list=[column for column in df.columns if column.endswith("log_rets")]
vol_list=[column for column in df.columns if column.endswith("vol")]
simple_rets_list=[column for column in df.columns if (column.endswith("log_rets")) and (column not in log_rets_list)]


# In[81]:


log_rets_list


# In[102]:


components=["USD_log_rets", "USD_gk_vol", "GGAL.BA_log_rets", "GGAL.BA_gk_vol"]
model=VAR(df[components])


# In[105]:





# In[106]:


results_df=pd.DataFrame(columns=["AIC", "BIC", "HQIC"])
models_dict={}
for i in range(1,
               int(np.round(12*(len(df.index)/100.)**(1./4), 0))): 
    # este es el valor que statsmodels define como standard si no se especifica maxlags
    result=model.fit(maxlags=i)
    models_dict[i]=model
    results_df.loc[i,"AIC"]=result.aic
    results_df.loc[i,"BIC"]=result.bic
    results_df.loc[i,"HQIC"]=result.hqic


# In[110]:


results_df


# In[114]:


order_select=model.select_order(int(np.round(12*(len(df.index)/100.)**(1./4), 0)))
order_select.summary()


# In[115]:


order_select.selected_orders


# In[ ]:


def plot_close_rets_vol(data, key, comp):
    model=models[f"{key}_{comp}_model"]
    prediction=models[f"{key}_{comp}_prediction"]
    states=set(prediction)

    fig=plt.figure(figsize = (20, 20))
    plt.tight_layout()
    plt.title(f"{key} Close, Log returns and intraday Vol\n{comp} states")

    for subplot, var in zip(range(1,4), ["Close", "log_rets", "gk_vol"]):    
        plt.subplot(3,1,subplot)
        for i in set(prediction):
            state = (prediction == i)
            x = data[key].index[state]
            y = data[key][var].iloc[state]
            plt.plot(x, y, '.')
        plt.legend(states, fontsize=16)
        
        plt.grid(True)
        plt.xlabel("datetime", fontsize=16)
        plt.ylabel(var, fontsize=16)
            
    plt.savefig(os.path.join(resultsroute, "graphs", 
                             f"{comp}_states", 
                             f"{key}_model_{comp}.png"))


# In[ ]:


for key in data.keys():
    for comp in comps:
        plot_close_rets_vol(data, key, comp)


# ## HMM Selection

# Selecting the Number of States in Hidden Markov Models: Pragmatic Solutions Illustrated Using Animal Movement
# https://sci-hub.st/10.1007/s13253-017-0283-8

# In[ ]:




