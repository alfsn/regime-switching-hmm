#!/usr/bin/env python
# coding: utf-8

# ## Startup

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pomegranate as pm
import torch
from scipy.special import logsumexp

import logging
import os
import pickle
import warnings


# In[2]:


from pomegranate.distributions import Normal
from pomegranate.hmm import DenseHMM


# In[3]:


random_state = 42
np.random.seed(random_state)
logging.captureWarnings(True)


# In[4]:


from scripts.params import get_params
from scripts.aux_functions import (
    generate_columns,
    save_as_pickle,
    get_all_results_matching,
    clean_modelname,
)

params = get_params()


# ## Data Retrieval

# In[5]:


dataroute = params["dataroute"]
resultsroute = params["resultsroute"]
dumproute = params["dumproute"]


# In[6]:


name = f'finaldf_train_{params["tablename"]}.pickle'
filename = os.path.join(dataroute, name)
with open(filename, "rb") as handle:
    df = pickle.load(handle)


# In[7]:


df.head()


# ## HMM Training

# In[8]:


range_states = range(1, 16)
emptydf = pd.DataFrame(columns=["AIC", "BIC"], index=range_states)
emptydf.fillna(np.inf, inplace=True)
results_dict_df = {stock: emptydf for stock in params["tickerlist"]}


# In[9]:


def from_df_to_reshaped(data: pd.DataFrame):
    npdata = data.values
    data_reshaped = npdata[:, :, np.newaxis]
    return data_reshaped


# In[10]:


def GaussianHMM(data_reshaped: np.ndarray, n_state: int):
    model = DenseHMM(distributions=[Normal() for _ in range(n_state)], sample_length=1)

    res = model.fit(data_reshaped)
    return res


# In[11]:


def n_params(res: pm.hmm.dense_hmm.DenseHMM):
    n_dist = res.n_distributions
    params_from_dists = n_dist * 2  # mean and variance for Normal
    transmat_elements = n_dist * (
        n_dist - 1
    )  # square matrix (minus last row bc must sum to one)
    n_params = params_from_dists + transmat_elements
    return n_params


# In[12]:


def get_aic(res: pm.hmm.dense_hmm.DenseHMM, data: np.ndarray):
    """
    Log Likelihood of the model is the Logsumexp of the log likelihood
    see https://stats.stackexchange.com/questions/60902/how-to-calculate-the-log-likelihood-in-hmm-from-the-output-of-the-forward-algori
    """
    aic = 2 * n_params(res) - 2 * logsumexp(res.log_probability(data))
    return aic


# In[13]:


def get_bic(res: pm.hmm.dense_hmm.DenseHMM, data: np.ndarray):
    """
    bic = k * np.log(len(data)) - 2 * model.log_likelihood(data)
    """
    bic = n_params(res) * np.log(len(data)) - 2 * logsumexp(res.log_probability(data))
    return bic


# In[14]:


def select_best(data: pd.DataFrame, max_states=15):

    aic = {"criterion": np.inf, "best_model": None, "n_state": None}
    bic = {"criterion": np.inf, "best_model": None, "n_state": None}

    data_reshaped = from_df_to_reshaped(data)

    for num_states in range(2, max_states + 1):
        res = GaussianHMM(data_reshaped, n_state=num_states)

        aic_result = get_aic(res, data_reshaped)
        bic_result = get_bic(res, data_reshaped)

        if aic_result < aic["criterion"]:
            aic["criterion"] = aic_result
            aic["best_model"] = res
            aic["n_state"] = num_states
        if bic_result < bic["criterion"]:
            bic["criterion"] = bic_result
            bic["best_model"] = res
            bic["n_state"] = num_states

    return aic, bic


# In[15]:


def find_best_all_assets(
    df: pd.DataFrame,
    max_states: int = 10,
    contains_vol: bool = False,
    contains_USD: bool = False,
):
    best = {stock: {"aic": None, "bic": None} for stock in params["assetlist"]}

    for stock in params["assetlist"]:
        print(stock)
        cols = generate_columns(
            stock=stock, contains_vol=contains_vol, contains_USD=contains_USD
        )
        aic, bic = select_best(df[cols], max_states=max_states)
        best[stock]["aic"] = aic
        best[stock]["bic"] = bic

    return best


# In[16]:


df[["USD_^BVSP_log_rets", "USD_^BVSP_gk_vol"]] = df[
    ["^BVSP_log_rets", "^BVSP_gk_vol"]
].copy()
# transitorio pq issue #71


# In[17]:


for i in range(5):
    try:
        best_with_vol = find_best_all_assets(
            df, max_states=10, contains_vol=True, contains_USD=False
        )
        # this cell sometimes crashes unexpectedly - just run again
        break
    except IndexError:
        print(f"Fail {i}, try again")
        


# In[18]:


for i in range(5):
    try:
        best_multiv = find_best_all_assets(
            df, max_states=10, contains_vol=True, contains_USD=True
        )
        # this cell sometimes crashes unexpectedly - just run again
        break
    except IndexError:
        print(f"Fail {i}, try again")


# In[19]:


best_multiv


# # Generating out of sample data

# In[20]:


name = f'finaldf_test_{params["tablename"]}.pickle'
filename = os.path.join(dataroute, name)
with open(filename, "rb") as handle:
    df_test = pickle.load(handle)


# In[21]:


df_test[["USD_^BVSP_log_rets", "USD_^BVSP_gk_vol"]] = df_test[
    ["^BVSP_log_rets", "^BVSP_gk_vol"]
].copy()
# transitorio pq issue #71


# In[22]:


def return_residuals(actual: pd.DataFrame, forecasts: pd.DataFrame):
    residuals = actual - forecasts
    return residuals


# In[23]:


def generate_samples_residuals(n_state, insample_data, oos_data):
    """
    This function only requires the number of normal distributions, which may be acquired from len(res.distributions)
    """
    # res.predict_proba(data_reshaped)[-1] es la matriz de cada estado
    columns = oos_data.columns

    split_date = oos_data.index[0]
    dates_to_forecast = len(oos_data.index)

    probabilities = pd.DataFrame(columns=range(n_state), index=oos_data.index)
    forecasts = pd.DataFrame(columns=oos_data.columns, index=oos_data.index)

    full_data = pd.concat([insample_data, oos_data])
    index = full_data.index
    end_loc = np.where(index >= split_date)[0].min()
    # esto es un int del iloc
    # preciso usar ints de iloc porque el timedelta se me va a romper con el fin de semana
    rolling_window = 252

    model_list = []

    for i in range(1, dates_to_forecast):
        # recursive window forecasting
        date_of_first_forecast = full_data.index[end_loc + i - 1]

        fitstart = end_loc - rolling_window + i
        fitend = end_loc + i

        # fit model with last year
        fit_data = full_data.iloc[fitstart:fitend][columns]
        reshaped_fit_data= from_df_to_reshaped(fit_data)
        
        res = GaussianHMM(data_reshaped=reshaped_fit_data, n_state=n_state)
        model_list.append(res)
        
        prob_matrix = res.predict_proba(reshaped_fit_data)[-1]
        prob_states = prob_matrix.sum(axis=0)/prob_matrix.sum() # rescale to measure 1
        
        last_day_state_probs = prob_matrix.sum(axis=0) / prob_matrix.sum()
        # hotfix véase https://github.com/alfsn/regime-switching-hmm/issues/72

        probabilities.loc[date_of_first_forecast] = last_day_state_probs
        
        param_means = [dist.means for dist in res.distributions]
        param_tensor = torch.cat(param_means, dim=0)

        expected_means = torch.dot(prob_states, param_tensor)
        
        forecasts.loc[date_of_first_forecast] = expected_means

    forecasts.fillna(method="ffill", inplace=True)

    residuals = return_residuals(oos_data, forecasts)

    return probabilities, forecasts, residuals
        


# In[24]:


def generate_and_save_samples(
    best_model_dict: dict,
    modeltype: str,
    insample_data: pd.DataFrame,
    oos_data: pd.DataFrame,
    contains_vol: bool,
    contains_USD: bool,
):
    generic_dict = {stock: None for stock in params["tickerlist"]}
    probabilities = {"aic": generic_dict.copy(), "bic": generic_dict.copy()}
    forecasts = probabilities.copy()
    residuals = probabilities.copy()

    for stock in best_model_dict.keys():
        for criterion, specific_model in best_model_dict[stock].items():
            retries=5
            n_state = specific_model["n_state"]
            print(modeltype, criterion, stock, n_state)
            columns = generate_columns(
                stock=stock, contains_vol=contains_vol, contains_USD=contains_USD
            )
            
            for i in range(retries):
                try:
                    proba, fcast, resid= generate_samples_residuals(
                        n_state=n_state,
                        insample_data=insample_data[columns],
                        oos_data=oos_data[columns],
                    )
                    print("Converged")
                    break
                except IndexError:
                    print(f"Fail {i}, retrying...")

            probabilities[criterion][stock] = proba
            forecasts[criterion][stock] = fcast
            residuals[criterion][stock] = resid

    for criterion in ["aic", "bic"]:
        save_as_pickle(
            data=forecasts[criterion],
            resultsroute=params["resultsroute"],
            model_type=f"HMM_{modeltype}",
            tablename=params["tablename"],
            criterion=criterion,
            type_save="forecasts",
        )

        save_as_pickle(
            data=residuals[criterion],
            resultsroute=params["resultsroute"],
            model_type=f"HMM_{modeltype}",
            tablename=params["tablename"],
            criterion=criterion,
            type_save="residuals",
        )


# In[25]:


models_dict = {
    "with_vol": (best_with_vol, True, False),
    "multiv": (best_multiv, True, True)
}


# In[26]:


for modeltype, tupla in models_dict.items():
    best_model_dict, contains_vol, contains_USD = tupla
    generate_and_save_samples(
        best_model_dict=best_model_dict,
        modeltype= modeltype,
        insample_data=df,
        oos_data=df_test,
        contains_vol= contains_vol,
        contains_USD=contains_USD)          

