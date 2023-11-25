#!/usr/bin/env python
# coding: utf-8

# ## Startup

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import aic, bic

import os
import pickle


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


np.random.seed(42)


# In[4]:


dataroute=os.path.join("..",  "data")
dumproute=os.path.join("..",  "dump")
resultsroute=os.path.join("..",  "results")


# ## Data Retrieval

# In[5]:


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


# In[6]:


with open(os.path.join(dumproute, "tickerlist.pickle"), 'rb') as f:
    tickerlist=pickle.load(f)
tickerlist


# In[7]:


df.head(3)


# # VAR Training

# In[8]:


log_rets_list=[column for column in df.columns if column.endswith("log_rets")]
vol_list=[column for column in df.columns if column.endswith("vol")]
simple_rets_list=[column for column in df.columns if (column.endswith("log_rets")) and (column not in log_rets_list)]


# In[9]:


emptydf=pd.DataFrame(columns=["AIC", "BIC"], index=range(1,11))
results_dict_df={stock:emptydf for stock in tickerlist}


# In[10]:


aic_best_model={stock:None for stock in tickerlist}
bic_best_model={stock:None for stock in tickerlist}

aic_best_residuals={stock:None for stock in tickerlist}
bic_best_residuals={stock:None for stock in tickerlist}


# In[11]:


for stock in tickerlist:
    columns = [f'{stock}_log_rets', f'{stock}_gk_vol']
    stock_data = df[columns]
    
    for lag in range(1, 11):
        model = VAR(stock_data)
        results = model.fit(lag)

        results_dict_df[stock].loc[lag, "AIC"]=results.aic
        results_dict_df[stock].loc[lag, "BIC"]=results.bic

    best_aic_lag=results_dict_df[stock]["AIC"].astype(float).idxmin()
    best_bic_lag=results_dict_df[stock]["BIC"].astype(float).idxmin()

    aic_best_model[stock]=VAR(stock_data).fit(best_aic_lag)
    bic_best_model[stock]=VAR(stock_data).fit(best_bic_lag)

    aic_best_residuals[stock]=aic_best_model[stock].resid
    bic_best_residuals[stock]=bic_best_model[stock].resid


# In[15]:


with open(os.path.join(resultsroute, "VAR_univ_aic_bestmodels.pickle"), "wb") as output_file:
    pickle.dump(aic_best_model, output_file)

with open(os.path.join(resultsroute, "VAR_univ_bic_bestmodels.pickle"), "wb") as output_file:
    pickle.dump(bic_best_model, output_file)


# In[16]:


with open(os.path.join(resultsroute, "VAR_univ_aic_residuals.pickle"), "wb") as output_file:
    pickle.dump(aic_best_residuals, output_file)

with open(os.path.join(resultsroute, "VAR_univ_bic_residuals.pickle"), "wb") as output_file:
    pickle.dump(bic_best_residuals, output_file)


# # with USD

# In[29]:


emptydf=pd.DataFrame(columns=["AIC", "BIC"], index=range(1,11))
results_dict_df={stock:emptydf for stock in tickerlist}


# In[30]:


aic_best_model={stock:None for stock in tickerlist}
bic_best_model={stock:None for stock in tickerlist}

aic_best_residuals={stock:None for stock in tickerlist}
bic_best_residuals={stock:None for stock in tickerlist}


# In[31]:


for stock in tickerlist:
    columns = ['USD_log_rets', 'USD_gk_vol', f'{stock}_log_rets', f'{stock}_gk_vol']
    stock_data = df[columns]
    
    for lag in range(1, 11):
        model = VAR(stock_data)
        results = model.fit(lag)

        results_dict_df[stock].loc[lag, "AIC"]=results.aic
        results_dict_df[stock].loc[lag, "BIC"]=results.bic

    best_aic_lag=results_dict_df[stock]["AIC"].astype(float).idxmin()
    best_bic_lag=results_dict_df[stock]["BIC"].astype(float).idxmin()

    aic_best_model[stock]=VAR(stock_data).fit(best_aic_lag)
    bic_best_model[stock]=VAR(stock_data).fit(best_bic_lag)

    aic_best_residuals[stock]=aic_best_model[stock].resid
    bic_best_residuals[stock]=bic_best_model[stock].resid


# In[32]:


with open(os.path.join(resultsroute, "VAR_multiv_aic_bestmodels.pickle"), "wb") as output_file:
    pickle.dump(aic_best_model, output_file)

with open(os.path.join(resultsroute, "VAR_multiv_bic_bestmodels.pickle"), "wb") as output_file:
    pickle.dump(bic_best_model, output_file)


# In[33]:


with open(os.path.join(resultsroute, "VAR_multiv_aic_residuals.pickle"), "wb") as output_file:
    pickle.dump(aic_best_residuals, output_file)

with open(os.path.join(resultsroute, "VAR_multiv_bic_residuals.pickle"), "wb") as output_file:
    pickle.dump(bic_best_residuals, output_file)


# In[34]:


bic_best_residuals["^MERV"]

