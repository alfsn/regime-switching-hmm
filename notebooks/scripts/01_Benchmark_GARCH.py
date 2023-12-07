#!/usr/bin/env python
# coding: utf-8

# ## Startup

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import arch

import os
import pickle


# In[3]:


np.random.seed(42)


# In[7]:


from scripts.params import get_params

params = get_params()


# In[4]:


dataroute=os.path.join("..",  "data")
processedroute=os.path.join("...", "processed")
resultsroute=os.path.join("..",  "results")


# ## Data Retrieval

# In[8]:


name=f'processed_train_{params["tablename"]}.pickle'
filename=os.path.join(dataroute, name)

with open(filename, 'rb') as handle:
    data=pickle.load(handle)


# ## GARCH Training

# In[9]:


# Define the range of p and q values
p_values = [1, 2, 3]  # Example: p values
q_values = [0, 1, 2, 3]  # Example: q values
# all models with q=0 are exclusively ARCH (non-GARCH)


# In[10]:


models = {}
predict = {}


# In[11]:


for key, ohlc_df in data.items():
    if ohlc_df["log_rets"].isna().any():
        print(key)
        #display(ohlc_df.loc[ohlc_df["log_rets"].isna()])


# In[12]:


best_aic={}
best_bic={}


# In[13]:


def check_best_aic(key, model, previous_best:float):
    """
    AIC is better when lower.
    """
    if model==None:
        pass
    else:
        if model.aic<previous_best:
            best_aic[key]=(model, model.aic)


# In[14]:


def check_best_bic(key, model, previous_best:float):
    """
    BIC is better when lower.
    """
    if model==None:
        pass
    else:
        if model.aic<previous_best:
            best_bic[key]=(model, model.bic)


# In[15]:


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

# In[17]:


aic_residuals={}
bic_residuals={}

for key in best_aic.keys():
    aic_residuals[key]=best_aic[key][0].resid
    bic_residuals[key]=best_bic[key][0].resid


# # Saving best models and residuals

# In[18]:


with open(os.path.join(resultsroute, f"""GARCH_{params["tablename"]}_aic_bestmodels.pickle"""), "wb") as output_file:
    pickle.dump(best_aic, output_file)

with open(os.path.join(resultsroute, f"""GARCH_{params["tablename"]}_bic_bestmodels.pickle"""), "wb") as output_file:
    pickle.dump(best_bic, output_file)


# Los modelos sirven los residuos NO!
# https://github.com/alfsn/regime-switching-hmm/issues/27

# In[19]:


with open(os.path.join(resultsroute, f"""GARCH_{params["tablename"]}_aic_residuals.pickle"""), "wb") as output_file:
    pickle.dump(aic_residuals, output_file)

with open(os.path.join(resultsroute, f"""GARCH_{params["tablename"]}_bic_residuals.pickle"""), "wb") as output_file:
    pickle.dump(bic_residuals, output_file)


# # Model prediction
# # NB this is currently unused and will only be used in the OOS part 
# 
# Function documentation: https://arch.readthedocs.io/en/latest/univariate/generated/generated/arch.univariate.base.ARCHModelResult.forecast.html#arch.univariate.base.ARCHModelResult.forecast

# In[20]:


for key, ohlc_df in data.items():
    for p in p_values:
        for q in q_values:
            for dist in ['Normal', 'StudentsT']:
                # Predictions on the training data
                pred = results.forecast()
                predict[key][(p, q, dist)] = predict


# # Plotting
# ## TODO: Esto aun está feo: tengo que armar que esto devuelva el plotteo de returns y los predicts uno encima del otro

# In[24]:


def plot_close_rets(data, model, key, name):
    fig=plt.figure(figsize = (20, 20))
    plt.tight_layout()
    plt.title(f"{key} Log returns")
    
    plt.subplot(1, 1, 1)

    x = data[key]["log_rets"]
    y = data[key].index
    
    plt.plot(x, y, '.', c="red")
    #plt.plot(x, model.predict(x), '.', c="blue")        
        
    plt.grid(True)
    plt.xlabel("datetime", fontsize=16)
    plt.ylabel("log rets", fontsize=16)
            
    plt.savefig(os.path.join(resultsroute, "graphs", 
                             f"GARCH", 
                             f"{key}_model_{name}.png"))


# In[26]:


for key in data.keys():
    print(key)
    plot_close_rets(data, key)
plt.show()


# In[ ]:




