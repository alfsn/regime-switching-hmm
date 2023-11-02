#!/usr/bin/env python
# coding: utf-8

# ## Startup

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, coint
from statsmodels.tools.eval_measures import rmse, aic

import os
import pickle


# In[2]:


np.random.seed(42)


# In[41]:


dataroute=os.path.join("..",  "data")
dumproute=os.path.join("..",  "dump")
resultsroute=os.path.join("..",  "results")


# ## Data Retrieval

# In[4]:


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


# In[5]:


df.head(3)


# # VAR Training

# ## Lag selection
# In this instance, we will select an optimal lag length for all models.
# We will take a single value for all VARs using GGAL.BA, since this is the stock that provides the highest volume in the Argentine Market.
# 
# If we require to fit every single var model to its optimum, we will use
# VAR().fit(maxlags=15, ic='aic')

# In[6]:


df.columns


# In[7]:


log_rets_list=[column for column in df.columns if column.endswith("log_rets")]
vol_list=[column for column in df.columns if column.endswith("vol")]
simple_rets_list=[column for column in df.columns if (column.endswith("log_rets")) and (column not in log_rets_list)]


# In[8]:


log_rets_list


# In[9]:


components=["USD_log_rets", "USD_gk_vol", "GGAL.BA_log_rets", "GGAL.BA_gk_vol"]
model=VAR(df[components])


# In[10]:


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


# In[11]:


results_df


# In[12]:


order_select=model.select_order(int(np.round(12*(len(df.index)/100.)**(1./4), 0)))
order_select.summary()


# In[13]:


order_select.selected_orders


# The selected order will be, by parsimony, 2 lags.

# In[16]:


orderlag=2


# In[22]:


# importamos la lista de variables
with open(os.path.join(dumproute, "tickerlist.pickle"), "rb") as f:
    tickerlist=pickle.load(f)
    
tickerlist


# In[38]:


var_models = {}
# Iterate over each stock
for stock in tickerlist:
    # Select relevant columns for the VAR model
    columns = ['USD_log_rets', 'USD_gk_vol', f'{stock}_log_rets', f'{stock}_gk_vol']
    # Create a new DataFrame for the VAR model
    var_data = df[columns]
    # Fit a VAR model for the current stock
    model = VAR(var_data)
    results = model.fit(orderlag)
    # Store the VAR model results in the dictionary
    var_models[stock] = results


# In[39]:


var_models["GGAL"].params


# In[43]:


with open(os.path.join(resultsroute, "VARdict.pickle"), "wb") as f:
    pickle.dump(var_models, f, pickle.HIGHEST_PROTOCOL)


# In[ ]:




