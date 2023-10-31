#!/usr/bin/env python
# coding: utf-8

# ## Startup

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arch

import os
import pickle


# In[5]:


np.random.seed(42)


# In[6]:


dataroute=os.path.join("..",  "data")
resultsroute=os.path.join("..",  "results")


# ## Data Retrieval

# In[7]:


start='2013-01-01'
end="2023-06-01"

name=f'processed_dataset_{start}_{end}.pickle'
filename=os.path.join(dataroute, name)

with open(filename, 'rb') as handle:
    data=pickle.load(handle)


# ## ARCH Training

# In[20]:


# Define the range of p and q values
p_values = [1, 2, 3]  # Example: p values
q_values = [0, 1, 2, 3]  # Example: q values


# In[21]:


models = {}
predict = {}


# In[31]:


# Estimate ARMA-ARCH and ARMA-GARCH models for different p and q values
for key, ohlc_df in data.items():
    returns = ohlc_df['log_rets']*100 
    # TODO: CHECKEAR ESTE RESCALING!!!
    
    models[key] = {}
    predict[key] = {}

    for p in p_values:
        for q in q_values:
            for dist in ['Normal', 'StudentsT']:
                print(p+q)
                model = arch.arch_model(returns, vol='Garch', p=p, q=q, dist=dist)
                results = model.fit()

                models[key][(p, q, dist)] = results


# # Model prediction
# 
# Function documentation: https://arch.readthedocs.io/en/latest/univariate/generated/generated/arch.univariate.base.ARCHModelResult.forecast.html#arch.univariate.base.ARCHModelResult.forecast

# In[ ]:


for key, ohlc_df in data.items():
    for p in p_values:
        for q in q_values:
            for dist in ['Normal', 'StudentsT']:
                # Predictions on the training data
                pred = results.forecast()
                predict[key][(p, q, dist)] = predict


# In[29]:


type(models["^MERV"][(1,0,"Normal")])


# In[ ]:


# Plotting


# In[23]:


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


# In[24]:


for key in data.keys():
    for comp in comps:
        plot_close_rets_vol(data, key, comp)
plt.show()

