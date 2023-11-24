#!/usr/bin/env python
# coding: utf-8

# ## Startup

# In[35]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import arch

import os
import pickle

import warnings
from sklearn.exceptions import ConvergenceWarning


# In[2]:


np.random.seed(42)


# In[73]:


dataroute=os.path.join("..",  "data")
processedroute=os.path.join("...", "processed")
resultsroute=os.path.join("..",  "results")


# ## Data Retrieval

# In[4]:


start='2013-01-01'
end="2023-06-01"

name=f'processed_dataset_{start}_{end}.pickle'
filename=os.path.join(dataroute, name)

with open(filename, 'rb') as handle:
    data=pickle.load(handle)


# ## GARCH Training

# In[5]:


# Define the range of p and q values
p_values = [1, 2, 3]  # Example: p values
q_values = [0, 1, 2, 3]  # Example: q values
# all models with q=0 are exclusively ARCH (non-GARCH)


# In[6]:


models = {}
predict = {}


# In[7]:


for key, ohlc_df in data.items():
    if ohlc_df["log_rets"].isna().any():
        print(key)
        #display(ohlc_df.loc[ohlc_df["log_rets"].isna()])


# In[67]:


best_aic={}
best_bic={}


# In[68]:


def check_best_aic(key, model, previous_best:float):
    """
    AIC is better when lower.
    """
    if model==None:
        pass
    else:
        if model.aic<previous_best:
            best_aic[key]=(model, model.aic)


# In[69]:


def check_best_bic(key, model, previous_best:float):
    """
    BIC is better when lower.
    """
    if model==None:
        pass
    else:
        if model.aic<previous_best:
            best_bic[key]=(model, model.bic)


# In[70]:


# Estimate ARMA-ARCH and ARMA-GARCH models for different p and q values
nonconverged_models=0
ok_models=0

for key, ohlc_df in data.items():
    returns = ohlc_df['log_rets']
    
    models[key] = {}
    predict[key] = {}

    best_aic[key]=(None, np.inf)
    best_bic[key]=(None, np.inf)

    for p in p_values:
        for q in q_values:
            for dist in ['Normal', 'StudentsT']:
                model = arch.arch_model(returns, 
                                        mean="AR",
                                        lags=1,
                                        vol='Garch', 
                                        p=p, q=q, dist=dist, 
                                        rescale=False)
                results = model.fit(options={"maxiter":2000}, 
                                        disp="off", 
                                        show_warning=False)

                if results.convergence_flag!=0:
                    # 0 is converged successfully
                    # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_slsqp.html
                    results=None                
                    nonconverged_models+=1
                else:
                    ok_models+=1
                
                check_best_aic(key=key, model=results, previous_best=best_aic[key][1])
                check_best_bic(key=key, model=results, previous_best=best_bic[key][1])

                models[key][(p, q, dist)] = results

print()
print(f"ok: {ok_models}")
print(f"nonconverged: {nonconverged_models}")


# # Residuals

# In[72]:


aic_residuals={}
bic_residuals={}

for key in best_aic.keys():
    aic_residuals[key]=best_aic[key][0].resid
    bic_residuals[key]=best_bic[key][0].resid


# # Saving best models and residuals

# In[74]:


with open(os.path.join(resultsroute, "GARCH_aic_bestmodels.pickle"), "wb") as output_file:
    pickle.dump(best_aic, output_file)

with open(os.path.join(resultsroute, "GARCH_bic_bestmodels.pickle"), "wb") as output_file:
    pickle.dump(best_bic, output_file)


# In[75]:


with open(os.path.join(resultsroute, "GARCH_aic_residuals.pickle"), "wb") as output_file:
    pickle.dump(aic_residuals, output_file)

with open(os.path.join(resultsroute, "GARCH_bic_residuals.pickle"), "wb") as output_file:
    pickle.dump(bic_residuals, output_file)


# In[ ]:





# # Model prediction
# # NB this is currently unused and will only be used in the OOS part 
# 
# Function documentation: https://arch.readthedocs.io/en/latest/univariate/generated/generated/arch.univariate.base.ARCHModelResult.forecast.html#arch.univariate.base.ARCHModelResult.forecast

# In[9]:


for key, ohlc_df in data.items():
    for p in p_values:
        for q in q_values:
            for dist in ['Normal', 'StudentsT']:
                # Predictions on the training data
                pred = results.forecast()
                predict[key][(p, q, dist)] = predict


# # Plotting

# In[10]:


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


# In[11]:


for key in data.keys():
    for comp in comps:
        plot_close_rets_vol(data, key, comp)
plt.show()

