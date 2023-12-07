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


random_state=42
np.random.seed(random_state)


# In[3]:


from scripts.params import get_params

params = get_params()


# ## Data Retrieval

# In[4]:


dataroute=os.path.join("..",  "data")
dumproute=os.path.join("..",  "dump")
resultsroute=os.path.join("..",  "results")


# In[25]:


name=f'finaldf_train_{params["tablename"]}.pickle'
filename=os.path.join(dataroute, name)
with open(filename, 'rb') as handle:
    df=pickle.load(handle)


# In[26]:


df.head()


# In[27]:


tickerlist=params["tickerlist"]


# ## HMM Training

# In[28]:


range_states=range(1,11)
emptydf=pd.DataFrame(columns=["AIC", "BIC"], index=range_states)
emptydf.fillna(np.inf)
results_dict_df={stock:emptydf for stock in tickerlist}


# In[29]:


aic_best_model={stock:None for stock in tickerlist}
bic_best_model={stock:None for stock in tickerlist}

aic_best_residuals={stock:None for stock in tickerlist}
bic_best_residuals={stock:None for stock in tickerlist}


# In[30]:


for stock in tickerlist:
    columns = [f'{stock}_log_rets', f'{stock}_gk_vol']
    insample_data = df[columns]

    param_dict={
        "covariance_type" : "diag", 
        "n_iter" : 500,
        "random_state" : random_state
        #no voy a usar startprob_prior por devlog 20-06-23
        }

    for nstate in range_states:
        print(stock, nstate)
        model = hmm.GaussianHMM(n_components= nstate, **param_dict, verbose=False)
        results = model.fit(insample_data)

        # TODO: aca tengo que poner un check de model convergence
        if results.monitor_.converged:
            try:
                results_dict_df[stock].loc[nstate, "AIC"]=model.aic(insample_data)
                results_dict_df[stock].loc[nstate, "BIC"]=model.bic(insample_data)
            except ValueError:
                pass


# In[32]:


for stock in tickerlist:
    columns = [f'{stock}_log_rets', f'{stock}_gk_vol']
    insample_data = df[columns]
    
    best_aic_nstate=results_dict_df[stock]["AIC"].astype(float).idxmin()
    best_bic_nstate=results_dict_df[stock]["BIC"].astype(float).idxmin()

    aic_best_model[stock]=hmm.GaussianHMM(n_components = best_aic_nstate, **param_dict).fit(insample_data)
    bic_best_model[stock]=hmm.GaussianHMM(n_components = best_bic_nstate, **param_dict).fit(insample_data)


# In[34]:


def generate_samples_residuals(model, insample_data, oos_data):
    """
    
    """
    # como el modelo es memoryless, sólo necesito 1 día de observación para saber en qué estado estoy
    # por lo tanto, en vez de complicarme con dos datasets, puedo agregarle el ultimo día de insample_data al ppio de oos_data
    # pseudocodigo
    oos_data=pd.concat([insample_data[-1:], oos_data])
    del insample_data

    samples=pd.DataFrame(columns=oos_data.columns)
    residuals=pd.DataFrame(columns=oos_data.columns)

    # for i=0
    for i in range(1,
                   len(oos_data.index)):
        prev_obs=oos_data[i-1:i]

        todays_obs = oos_data[i:i+1]
        todays_date = todays_obs.index

        state=model.decode(prev_obs)[1][-1]
        # decode()[0] is the log probability, decode()[1] is the sequence of states, [-1] is the last state
        # since we have added the last datum of insample_data to oos_data, then the 
            # TODO: revisar que tenga sentido decodear solo el ultimo día.
            # La alternativa es agregar diez días de insample al principio y usar un decode con diez dias, 
            # me quedo con el ultimo valor del array que maximiza la log-likelihood de la secuencia entera
            # pero como es memoryless, not sure if it makesense
        
        sample = model.sample(n_samples=1, random_state=random_state, currstate=state)[0]
        # sample()[0] is the array with observations of the sampled variables, sample()[1] is the value of the currstate
        sample = pd.DataFrame(data=sample, columns=oos_data.columns, index=todays_date)

        samples=pd.concat([samples, sample])   
        # sampling given state t-1
        # observar realización en t+i
        residual = todays_obs-sample
        
        residuals=pd.concat([residuals, residual])
    
    return samples, residuals


# In[36]:


samples, residuals = generate_samples_residuals(aic_best_model[params["index"]], insample_data=insample_data, oos_data=insample_data)


# In[37]:


residuals


# # Guardado de datos

# In[38]:


with open(os.path.join(resultsroute, f"""HMM_univ_{params["tablename"]}_aic_bestmodels.pickle"""), "wb") as output_file:
    pickle.dump(aic_best_model, output_file)

with open(os.path.join(resultsroute, f"""HMM_univ_{params["tablename"]}_bic_bestmodels.pickle"""), "wb") as output_file:
    pickle.dump(bic_best_model, output_file)


# Los modelos sirven los residuos NO!
# https://github.com/alfsn/regime-switching-hmm/issues/27

# In[39]:


with open(os.path.join(resultsroute, f"""HMM_univ_{params["tablename"]}_aic_residuals.pickle"""), "wb") as output_file:
    pickle.dump(aic_best_residuals, output_file)

with open(os.path.join(resultsroute, f"""HMM_univ_{params["tablename"]}_bic_residuals.pickle"""), "wb") as output_file:
    pickle.dump(bic_best_residuals, output_file)


# In[ ]:





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
                                          n_iter = 500,
                                          random_state = random_state)
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


# In[ ]:


for key in data.keys():
    for comp in comps:
        plot_close_rets_vol(data, key, comp)


# ## Coefficients 

# In[9]:


models.keys()


# In[10]:


for model in models.keys()
    print(model)
    print("Matriz de transicion:")
    print(model.transmat_)
    print("Matriz de emisiones:")
    print(model.means_)


# ## HMM Selection

# Selecting the Number of States in Hidden Markov Models: Pragmatic Solutions Illustrated Using Animal Movement
# https://sci-hub.st/10.1007/s13253-017-0283-8

# In[ ]:




