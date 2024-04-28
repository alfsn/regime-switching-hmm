#!/usr/bin/env python
# coding: utf-8

# # Comparison
# 

# In[4]:


import pandas as pd
import numpy as np
import os
import pickle

pd.set_option("display.max_columns", None)


# In[6]:


from scripts.params import get_params
params = get_params()


# In[7]:


dataroute = params["dataroute"]
resultsroute = params["resultsroute"]
dumproute = params["dumproute"]
graphsroute = params["graphsroute"]


# In[8]:


forecasts="forecasts_by_stock_" + params["tablename"] + ".pickle"
filename=os.path.join(resultsroute, forecasts)
with open(filename, "rb") as handle:
    forecasts_by_stock = pickle.load(handle)


# In[10]:


residuals="residuals_by_stock_" + params["tablename"] + ".pickle"
filename=os.path.join(resultsroute, forecasts)
with open(filename, "rb") as handle:
    residuals_by_stock = pickle.load(handle)

