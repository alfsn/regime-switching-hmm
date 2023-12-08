#!/usr/bin/env python
# coding: utf-8

# ## Startup

# In[44]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm

import os
import pickle


# In[45]:


random_state=42
np.random.seed(random_state)


# In[46]:


from scripts.params import get_params

params = get_params()


# ## Data Retrieval

# In[47]:


dataroute=os.path.join("..",  "data")
dumproute=os.path.join("..",  "dump")
resultsroute=os.path.join("..",  "results")


# In[48]:


name=f'finaldf_train_{params["tablename"]}.pickle'
filename=os.path.join(dataroute, name)
with open(filename, 'rb') as handle:
    df=pickle.load(handle)


# In[49]:


df.head()


# In[50]:


tickerlist=params["tickerlist"]


# ## HMM Training

# In[51]:


range_states=range(1,16)
emptydf=pd.DataFrame(columns=["AIC", "BIC"], index=range_states)
emptydf.fillna(np.inf, inplace=True)
results_dict_df={stock:emptydf for stock in tickerlist}


# In[52]:


aic_best_model={stock:None for stock in tickerlist}
bic_best_model={stock:None for stock in tickerlist}


# In[53]:


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
        model = hmm.GaussianHMM(n_components= nstate, **param_dict, verbose=False)
        results = model.fit(insample_data)

        convergence=results.monitor_.converged
        # esta es la condición de si el modelo convergió

        all_states_found=np.isclose(a=(model.transmat_.sum(axis=1)), b=1).all()
        # esta es la condición de que todos los estados (nstates) hayan sido observados
        # si no, alguna fila en la matriz de transición del modelo son 0.
        # el errormsg es "Some rows of transmat_ have zero sum because no transition from the state was ever observed".

        startprob_check = (model.startprob_.sum()==1)
        # esta es la condición de que los estados al inicializar estén definidos
        
        good_model = convergence and all_states_found and startprob_check

        if good_model:
            try:
                results_dict_df[stock].loc[nstate, "AIC"]=model.aic(insample_data)
                results_dict_df[stock].loc[nstate, "BIC"]=model.bic(insample_data)
            except ValueError:
                pass
        else: 
            print(">"*10,f"{stock} {nstate} did not converge")
            results_dict_df[stock].loc[nstate, "BIC"]=np.inf
            results_dict_df[stock].loc[nstate, "BIC"]=np.inf


# In[54]:


for stock in tickerlist:
    columns = [f'{stock}_log_rets', f'{stock}_gk_vol']
    insample_data = df[columns]
    
    best_aic_nstate=results_dict_df[stock]["AIC"].astype(float).idxmin()
    best_bic_nstate=results_dict_df[stock]["BIC"].astype(float).idxmin()
    print(f"For stock {stock}, best AIC: {best_aic_nstate} best BIC: {best_bic_nstate}")

    aic_best_model[stock]=hmm.GaussianHMM(n_components = best_aic_nstate, **param_dict).fit(insample_data)
    bic_best_model[stock]=hmm.GaussianHMM(n_components = best_bic_nstate, **param_dict).fit(insample_data)


# # Generating out of sample data

# In[55]:


name=f'finaldf_test_{params["tablename"]}.pickle'
filename=os.path.join(dataroute, name)
with open(filename, 'rb') as handle:
    df_test=pickle.load(handle)


# In[56]:


def generate_HMM_samples_residuals(model, insample_data, oos_data):
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


# In[57]:


aic_best_residuals={stock:None for stock in tickerlist}
bic_best_residuals={stock:None for stock in tickerlist}


# In[60]:


for stock in tickerlist:
    columns = [f'{stock}_log_rets', f'{stock}_gk_vol']
    insample_data = df[columns]
    oos_data=df_test[columns]

    samples, aic_best_residuals[stock] = generate_HMM_samples_residuals(
        aic_best_model[stock], 
        insample_data=insample_data, 
        oos_data=oos_data)

    samples, bic_best_residuals[stock] = generate_HMM_samples_residuals(
        bic_best_model[stock], 
        insample_data=insample_data, 
        oos_data=oos_data)


# # Guardado de datos

# In[62]:


with open(os.path.join(resultsroute, f"""HMM_univ_{params["tablename"]}_aic_bestmodels.pickle"""), "wb") as output_file:
    pickle.dump(aic_best_model, output_file)

with open(os.path.join(resultsroute, f"""HMM_univ_{params["tablename"]}_bic_bestmodels.pickle"""), "wb") as output_file:
    pickle.dump(bic_best_model, output_file)


# In[63]:


with open(os.path.join(resultsroute, f"""HMM_univ_{params["tablename"]}_aic_residuals.pickle"""), "wb") as output_file:
    pickle.dump(aic_best_residuals, output_file)

with open(os.path.join(resultsroute, f"""HMM_univ_{params["tablename"]}_bic_residuals.pickle"""), "wb") as output_file:
    pickle.dump(bic_best_residuals, output_file)


# # Graficando

# In[72]:


def plot_close_rets_vol(model, data, key, IC):
    prediction= model.predict(data)
    states=set(prediction)

    fig=plt.figure(figsize = (20, 20))
    plt.tight_layout()
    plt.title(f"{key} Log returns and intraday Vol\n{model.n_components} states / best by {IC}")

    for subplot, var in zip(range(1,3), data.columns):    
        plt.subplot(2,1,subplot)
        for i in set(prediction):
            state = (prediction == i)
            x = data.index[state]
            y = data[var].iloc[state]
            plt.plot(x, y, '.')
        plt.legend(states, fontsize=16)
        
        plt.grid(True)
        plt.xlabel("datetime", fontsize=16)
        plt.ylabel(var, fontsize=16)
            
    plt.savefig(os.path.join(resultsroute, "graphs", 
                             f"HMM", 
                             f"{key}_model_{IC}.png"))


# In[ ]:


for dictionary, IC in zip([aic_best_model, bic_best_model], ["AIC", "BIC"]):
    for key, model in dictionary.items():
        columns = [f'{stock}_log_rets', f'{stock}_gk_vol']
        insample_data = df[columns]
        oos_data=df_test[columns]
        train_end=insample_data.index.max()
        data=pd.concat([insample_data, oos_data])

        plot_close_rets_vol(model, data, key, IC)


# ## HMM Selection

# Selecting the Number of States in Hidden Markov Models: Pragmatic Solutions Illustrated Using Animal Movement
# https://sci-hub.st/10.1007/s13253-017-0283-8
