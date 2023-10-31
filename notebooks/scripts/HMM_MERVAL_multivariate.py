#!/usr/bin/env python
# coding: utf-8

# ## Startup

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm

import os
import pickle


# In[2]:


np.random.seed(42)


# In[3]:


dataroute=os.path.join("..",  "data")
resultsroute=os.path.join("..",  "results")


# ## Data Retrieval

# In[4]:


start='2013-01-01'
end="2023-06-01"

name=f'processed_dataset_{start}_{end}.pickle'
filename=os.path.join(dataroute, name)


with open(filename, 'rb') as handle:
    data=pickle.load(handle)


# ## HMM Training

# In[5]:


models={}
comps=[2,3,4]
datacols=["log_rets","gk_vol"]

for key in data.keys():
    print(key)
    for comp in comps:
        print(comp)
        # TODO: Change from single modelname to [key][comp] nested dict
        modelname=f"{key}_{comp}_model"
        predictionname=f"{key}_{comp}_prediction"
        
        X = data[key][datacols].values.reshape(-1, len(datacols))
        # bivariate
        # log returns and intraday volatility        
        models[modelname]=hmm.GaussianHMM(n_components = comp, #no voy a usar startprob_prior por devlog 20-06-23
                                          covariance_type = "diag", 
                                          n_iter = 50,
                                          random_state = 42)
        models[modelname].fit(X)
        models[predictionname]=models[modelname].predict(X)


# In[6]:


# Predict the hidden states corresponding to observed X.
for key in data.keys():
    for comp in comps:
        print(">"*30, key)
        model=models[f"{key}_{comp}_model"]
        prediction=models[f"{key}_{comp}_prediction"]
        print("unique states: ", pd.unique(prediction))
        print("\nStart probabilities:")
        print(model.startprob_)
        print("\nTransition matrix:")
        print(model.transmat_)
        print("\nGaussian distribution means:")
        print(model.means_)
        print("\nGaussian distribution covariances:")
        print(model.covars_)
        print()


# In[7]:


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


# In[8]:


for key in data.keys():
    for comp in comps:
        plot_close_rets_vol(data, key, comp)


# ## HMM Selection

# Selecting the Number of States in Hidden Markov Models: Pragmatic Solutions Illustrated Using Animal Movement
# https://sci-hub.st/10.1007/s13253-017-0283-8

# In[ ]:




