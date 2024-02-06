#!/usr/bin/env python
# coding: utf-8

# # Comparison
# 

# In[1]:


import pandas as pd
import numpy as np
import os

from statsmodels.tools.eval_measures import mse, meanabs, medianabs

pd.set_option('display.max_columns', None)


# In[2]:


from scripts.params import get_params

params = get_params()


# In[3]:


dataroute=os.path.join("..",  "data")
dumproute=os.path.join("..",  "dump")
resultsroute=os.path.join("..",  "results")


# In[4]:


all_residuals = {}

for filename in os.listdir(resultsroute):
    file_path = os.path.join(resultsroute, filename)
    if os.path.isfile(file_path) and 'residual' in filename:
        all_residuals[filename] = file_path

print(all_residuals)


# In[5]:


residual_df = pd.DataFrame()

for name, dir in all_residuals.items():
    dict_with_dfs = pd.read_pickle(dir)
    print(name)
    for stock in dict_with_dfs.keys():
        if type(dict_with_dfs[stock])==pd.Series: 
            # univariate models are saved as series
            df=pd.DataFrame(dict_with_dfs[stock])
        else:
            try:
                # multivariate models are saved as dataframes
                df=pd.DataFrame(dict_with_dfs[stock][f"{stock}_log_rets"])
            except: # TODO: SACAR ESTO! Es un chanchullo pq hay algunas que son guardadas como None
                pass

        modelname=name.replace("residuals.pickle", "").replace(params["tablename"], "").replace("__", "_").replace("__", "_")
        df.columns=[modelname+"_"+stock]
        residual_df = pd.merge(residual_df, df, left_index=True, right_index=True, how='outer')


# In[6]:


residual_df.tail(4)


# In[7]:


#estadisticos de nans
(residual_df.isna().sum()/len(residual_df.index)*100).describe()


# In[ ]:


asset="GGAL"
for column in df.columns:
    if column.endswith(asset)


# In[9]:


params["assetlist"]


# In[ ]:




