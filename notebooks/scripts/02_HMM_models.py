#!/usr/bin/env python
# coding: utf-8

# ## Startup

# In[58]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm

import logging
import os
import pickle


# In[61]:


logging.captureWarnings(True)
hmm.logging.disable(level=80)


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


# In[5]:


name=f'finaldf_train_{params["tablename"]}.pickle'
filename=os.path.join(dataroute, name)
with open(filename, 'rb') as handle:
    df=pickle.load(handle)


# In[6]:


df.head()


# In[7]:


tickerlist=params["tickerlist"]


# ## HMM Training

# In[8]:


range_states=range(1,16)
emptydf=pd.DataFrame(columns=["AIC", "BIC"], index=range_states)
emptydf.fillna(np.inf, inplace=True)
results_dict_df={stock:emptydf for stock in tickerlist}


# In[9]:


aic_best_model={stock:None for stock in tickerlist}
bic_best_model={stock:None for stock in tickerlist}


# In[62]:


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


# In[63]:


for stock in tickerlist:
    columns = [f'{stock}_log_rets', f'{stock}_gk_vol']
    insample_data = df[columns]
    
    best_aic_nstate=results_dict_df[stock]["AIC"].astype(float).idxmin()
    best_bic_nstate=results_dict_df[stock]["BIC"].astype(float).idxmin()
    print(f"For stock {stock}, best AIC: {best_aic_nstate} best BIC: {best_bic_nstate}")

    aic_best_model[stock]=hmm.GaussianHMM(n_components = best_aic_nstate, **param_dict).fit(insample_data)
    bic_best_model[stock]=hmm.GaussianHMM(n_components = best_bic_nstate, **param_dict).fit(insample_data)


# # Generating out of sample data

# In[64]:


name=f'finaldf_test_{params["tablename"]}.pickle'
filename=os.path.join(dataroute, name)
with open(filename, 'rb') as handle:
    df_test=pickle.load(handle)


# In[65]:


def return_residuals(actual:pd.DataFrame, forecasts:pd.DataFrame):
    residuals = (actual - forecasts).dropna()
    return residuals


# In[66]:


def generate_HMM_samples_residuals(model, insample_data, oos_data):
    """_summary_

    Args:
        model (_type_): _description_
        insample_data (_type_): _description_
        oos_data (_type_): _description_
    """
    # pseudocodigo
    # agarra el mejor modelo (esto con una cantidad optima de params ya esta)
    # fittear t-j con t-j-252d
    # Darle un año de datos hasta t-j para que me prediga la secuencia (probabilidad) de estados.
        # Le pido que me prediga las probabilidades de cada estado durante el periodo t-j, t-j-252: 
        # esto me da una matriz de (252 x n estados)
        # esto entiendo es https://hmmlearn.readthedocs.io/en/latest/api.html#hmmlearn.hmm.GaussianHMM.predict_proba
    # Tomo la ultima fila de la matriz
    # Multiplico esa por el vector de medias estimadas: este punto es mi forecast. 
        # esto es model.means_ (!)    
    nstate=model.n_components
    columns=oos_data.columns

    split_date = oos_data.index[0]
    dates_to_forecast = len(oos_data.index)

    probabilities=pd.DataFrame(columns=range(nstate), index=oos_data.index)
    forecasts=pd.DataFrame(columns=oos_data.columns, index=oos_data.index)

    full_data = pd.concat([insample_data, oos_data])
    del insample_data

    # vamos a implementar recursive window forecasting
    
    index = full_data.index
    end_loc = np.where(index >= split_date)[0].min()
    # esto es un int del iloc
    # preciso usar ints de iloc porque el timedelta se me va a romper con el fin de semana
    rolling_window = 252

    nstate=model.n_components
    model = hmm.GaussianHMM(n_components = nstate, **param_dict, verbose=False)

    model_list=[]
    counter=0

    for i in range(1, dates_to_forecast):
        date_of_first_forecast = full_data.index[end_loc + i -1]
        
        fitstart = end_loc - rolling_window + i
        fitend = end_loc + i

        # fit model with last year
        fit_data=full_data.iloc[fitstart:fitend][columns]
        res=model.fit(fit_data)
        model_list.append(res)
        # TODO: que pasa si fittea mal?
        
        # obtenemos las probabilidades por estado del ultimo dia
        # son las probabilidades que maximizan la log/likelihood de toda la secuencia
        index=len(model_list)
        while index>0:
            try:
                add_count=False
                last_day_state_probs = res.predict_proba(fit_data)[-1]
                probabilities.loc[date_of_first_forecast] = last_day_state_probs
                index=0

            except ValueError:
                # this happens when startprob_ must sum to 1 (got nan)
                # si el modelo falla en el predict_proba, se utiliza el de t-1
                add_count=True
                index=index-1
                res=model_list[index]
                
        if add_count:
            counter=counter+1
        # model.means_ es es la media condicional a cada estado
            # cada columna representa cada columna del dataset
            # cada fila es un estado
        # el producto punto entre este y las probabilidades del ultimo día me da la media esperada por cada columna
        expected_means = np.dot(last_day_state_probs, model.means_)
        forecasts.loc[date_of_first_forecast]=expected_means
    
    residuals = return_residuals(oos_data, forecasts)

    print("failed models: ", counter)
    return probabilities, forecasts, residuals


# In[67]:


aic_best_probabilities={stock:None for stock in tickerlist}
bic_best_probabilities={stock:None for stock in tickerlist}

aic_best_forecast={stock:None for stock in tickerlist}
bic_best_forecast={stock:None for stock in tickerlist}

aic_best_residuals={stock:None for stock in tickerlist}
bic_best_residuals={stock:None for stock in tickerlist}


# In[68]:


for stock in aic_best_model.keys():
    print(stock)
    columns=[f"{stock}_log_rets", 
             f"{stock}_gk_vol"]
    probabilities, forecasts, residuals = generate_HMM_samples_residuals(aic_best_model[stock],
                                                                         insample_data=df[columns], 
                                                                         oos_data=df_test[columns])
    aic_best_probabilities[stock]=probabilities
    aic_best_probabilities[stock]=forecasts
    aic_best_probabilities[stock]=residuals
    
    probabilities, forecasts, residuals = generate_HMM_samples_residuals(bic_best_model[stock],
                                                                         insample_data=df[columns], 
                                                                         oos_data=df_test[columns])
    bic_best_probabilities[stock]=probabilities
    bic_best_probabilities[stock]=forecasts
    bic_best_probabilities[stock]=residuals
    print()


# # Guardado de datos

# In[70]:


with open(os.path.join(resultsroute, f"""HMM_univ_{params["tablename"]}_aic_bestmodels.pickle"""), "wb") as output_file:
    pickle.dump(aic_best_model, output_file)

with open(os.path.join(resultsroute, f"""HMM_univ_{params["tablename"]}_bic_bestmodels.pickle"""), "wb") as output_file:
    pickle.dump(bic_best_model, output_file)


# In[71]:


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


# # With USD

# In[ ]:




