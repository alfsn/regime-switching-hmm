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


# In[5]:


from scripts.params import get_params

params = get_params()


# ## Data Retrieval

# In[6]:


name=f"""processed_train_{params["tablename"]}.pickle"""
filename=os.path.join(dataroute, name)
with open(filename, 'rb') as handle:
    data=pickle.load(handle)
    
name=f"""finaldf_train_{params["tablename"]}.pickle"""
filename=os.path.join(dataroute, name)
with open(filename, 'rb') as handle:
    df=pickle.load(handle)


# In[7]:


tickerlist=params["tickerlist"]


# In[8]:


df.head(3)


# # VAR Training

# In[9]:


emptydf=pd.DataFrame(columns=["AIC", "BIC"], index=range(1,11))
results_dict_df={stock:emptydf for stock in tickerlist}


# In[10]:


aic_best_model={stock:None for stock in tickerlist}
bic_best_model={stock:None for stock in tickerlist}

aic_best_lags={stock:None for stock in tickerlist}
bic_best_lags={stock:None for stock in tickerlist}

aic_best_residuals={stock:None for stock in tickerlist}
bic_best_residuals={stock:None for stock in tickerlist}


# In[11]:


def create_select_best_VAR(stock:str, stock_data:pd.DataFrame):
    columns = [f'{stock}_log_rets', f'{stock}_gk_vol']
    stock_data = df[columns]
    
    for lag in range(1, 11):
        model = VAR(stock_data)
        results = model.fit(lag)

        results_dict_df[stock].loc[lag, "AIC"]=results.aic
        results_dict_df[stock].loc[lag, "BIC"]=results.bic

    best_aic_lag=results_dict_df[stock]["AIC"].astype(float).idxmin()
    best_bic_lag=results_dict_df[stock]["BIC"].astype(float).idxmin()

    aic_best_lags[stock]=best_aic_lag
    bic_best_lags[stock]=best_bic_lag

    aic_best_model[stock]=VAR(stock_data).fit(best_aic_lag)
    bic_best_model[stock]=VAR(stock_data).fit(best_bic_lag)


# In[12]:


for stock in tickerlist:
    columns = [f'{stock}_log_rets', f'{stock}_gk_vol']
    stock_data = df[columns]
    create_select_best_VAR(stock, stock_data)


# In[13]:


with open(os.path.join(resultsroute, f"""VAR_univ_{params["tablename"]}_aic_bestmodels.pickle"""), "wb") as output_file:
    pickle.dump(aic_best_model, output_file)

with open(os.path.join(resultsroute, f"""VAR_univ_{params["tablename"]}_bic_bestmodels.pickle"""), "wb") as output_file:
    pickle.dump(bic_best_model, output_file)


# # Generating residuals

# In[14]:


name=f'finaldf_test_{params["tablename"]}.pickle'
filename=os.path.join(dataroute, name)
with open(filename, 'rb') as handle:
    df_test=pickle.load(handle)


# In[15]:


# TODO: comparar mismas cantidades de información
# https://github.com/alfsn/regime-switching-hmm/issues/38


# In[16]:


def generate_VAR_samples_residuals(lags, insample_data, oos_data):
        # pseudocodigo
    # agarra el mejor modelo (esto con una cantidad optima de params ya esta)
    # k = cantidad de params
    # fittear t-j con t-j-252d
    split_date = oos_data.index[0]
    dates_to_forecast = len(oos_data.index)

    oos_data = pd.concat([insample_data, oos_data])
    del insample_data
    
    index = oos_data.index
    end_loc = np.where(index >= split_date)[0].min()

    rolling_window = 252

    residuals=pd.DataFrame()

    for i in range(1, dates_to_forecast):        
        fitstart = end_loc - rolling_window + i
        fitend = end_loc + i

        stock_data = oos_data.iloc[fitstart:fitend]

        model = VAR(stock_data)
        results = model.fit(lags)

        resid = results.resid.iloc[-1:]
        residuals = pd.concat([residuals, resid], axis=0)
    return residuals


# In[17]:


for stock in aic_best_lags.keys():
    columns=[f"{stock}_log_rets", f"{stock}_gk_vol"]
    aic_best_residuals[stock]=generate_VAR_samples_residuals(lags=aic_best_lags[stock], 
                                                             insample_data=df[columns], 
                                                             oos_data=df_test[columns])
    bic_best_residuals[stock]=generate_VAR_samples_residuals(lags=bic_best_lags[stock], 
                                                             insample_data=df[columns], 
                                                             oos_data=df_test[columns])
    


# In[18]:


with open(os.path.join(resultsroute, f"""VAR_univ_{params["tablename"]}_aic_residuals.pickle"""), "wb") as output_file:
    pickle.dump(aic_best_residuals, output_file)

with open(os.path.join(resultsroute, f"""VAR_univ_{params["tablename"]}_bic_residuals.pickle"""), "wb") as output_file:
    pickle.dump(bic_best_residuals, output_file)


# # with USD

# In[14]:


emptydf=pd.DataFrame(columns=["AIC", "BIC"], index=range(1,11))
results_dict_df={stock:emptydf for stock in tickerlist}


# In[15]:


aic_best_model={stock:None for stock in tickerlist}
bic_best_model={stock:None for stock in tickerlist}

aic_best_residuals={stock:None for stock in tickerlist}
bic_best_residuals={stock:None for stock in tickerlist}


# In[19]:


for stock in tickerlist:
    columns = ['USD_log_rets', 'USD_gk_vol', f'{stock}_log_rets', f'{stock}_gk_vol']
    stock_data = df[columns]
    create_select_best_VAR(stock, stock_data)


# In[20]:


with open(os.path.join(resultsroute, f"""VAR_multiv_{params["tablename"]}_aic_bestmodels.pickle"""), "wb") as output_file:
    pickle.dump(aic_best_model, output_file)

with open(os.path.join(resultsroute, f"""VAR_multiv_{params["tablename"]}_bic_bestmodels.pickle"""), "wb") as output_file:
    pickle.dump(bic_best_model, output_file)


# In[21]:


for stock in aic_best_lags.keys():
    columns = ['USD_log_rets', 'USD_gk_vol', f'{stock}_log_rets', f'{stock}_gk_vol']
    aic_best_residuals[stock]=generate_VAR_samples_residuals(lags=aic_best_lags[stock], 
                                                             insample_data=df[columns], 
                                                             oos_data=df_test[columns])
    bic_best_residuals[stock]=generate_VAR_samples_residuals(lags=bic_best_lags[stock], 
                                                             insample_data=df[columns], 
                                                             oos_data=df_test[columns])


# In[22]:


with open(os.path.join(resultsroute, f"""VAR_multiv_{params["tablename"]}_aic_residuals.pickle"""), "wb") as output_file:
    pickle.dump(aic_best_residuals, output_file)

with open(os.path.join(resultsroute, f"""VAR_multiv_{params["tablename"]}_bic_residuals.pickle"""), "wb") as output_file:
    pickle.dump(bic_best_residuals, output_file)

