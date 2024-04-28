#!/usr/bin/env python
# coding: utf-8

# # Comparison
# 

# In[2]:


import pandas as pd
import numpy as np
import os
import pickle

pd.set_option("display.max_columns", None)


# In[9]:


from scripts.params import get_params
params = get_params()

from scripts.aux_functions import subset_of_columns


# In[4]:


dataroute = params["dataroute"]
resultsroute = params["resultsroute"]
dumproute = params["dumproute"]
graphsroute = params["graphsroute"]


# In[5]:


forecasts="forecasts_by_stock_" + params["tablename"] + ".pickle"
filename=os.path.join(resultsroute, forecasts)
with open(filename, "rb") as handle:
    forecasts_by_stock = pickle.load(handle)


# In[6]:


residuals="residuals_by_stock_" + params["tablename"] + ".pickle"
filename=os.path.join(resultsroute, forecasts)
with open(filename, "rb") as handle:
    residuals_by_stock = pickle.load(handle)


# In[10]:


subset_of_columns(subset_of_columns(residuals_by_stock["VALE"], "HMM"), "aic")


# In[ ]:


def plot_close_rets(data, model, key, name):
    fig = plt.figure(figsize=(20, 20))
    plt.tight_layout()
    plt.title(f"{key} Log returns")

    plt.subplot(1, 1, 1)

    x = data[key]["log_rets"]
    y = data[key].index

    plt.plot(x, y, ".", c="red")
    # plt.plot(x, model.predict(x), '.', c="blue")

    plt.grid(True)
    plt.xlabel("datetime", fontsize=16)
    plt.ylabel("log rets", fontsize=16)

    plt.savefig(
        os.path.join(resultsroute, "graphs", f"GARCH", f"{key}_model_{name}.png")
    )

