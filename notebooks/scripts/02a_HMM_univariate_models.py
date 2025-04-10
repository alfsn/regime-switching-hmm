#!/usr/bin/env python
# coding: utf-8

# ## Startup

# In[37]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm

import logging
import os
import pickle
import warnings


# In[38]:


logging.captureWarnings(True)
hmm.logging.disable(level=80)


# In[39]:


random_state = 42
np.random.seed(random_state)


# In[40]:


from scripts.params import get_params
from scripts.aux_functions import generate_columns, save_as_pickle, get_all_results_matching, clean_modelname

params = get_params()


# ## Data Retrieval

# In[41]:


dataroute = params["dataroute"]
resultsroute = params["resultsroute"]
dumproute = params["dumproute"]


# In[42]:


name = f'finaldf_train_{params["tablename"]}.pickle'
filename = os.path.join(dataroute, name)
with open(filename, "rb") as handle:
    df = pickle.load(handle)


# ## HMM Training

# In[43]:


range_states = range(1, 11)
emptydf = pd.DataFrame(columns=["AIC", "BIC"], index=range_states)
emptydf.fillna(np.inf, inplace=True)
results_dict_df = {stock: emptydf for stock in params["assetlist"]}


# In[44]:


param_dict = {
    "covariance_type": "diag",
    "n_iter": 500,
    "random_state": random_state,
    # no voy a usar startprob_prior por devlog 20-06-23
}


# In[45]:


def fit_hmm_model(
    df: pd.DataFrame,
    tickerlist: list,
    range_states,
    param_dict: dict,
    contains_vol: bool,
    contains_USD: bool,
):

    results_dict_df = {}

    for stock in params["assetlist"]:
        results_dict_df[stock] = pd.DataFrame(
            index=range_states, columns=["AIC", "BIC"]
        )
        for nstate in range_states:
            columns = generate_columns(stock, contains_vol, contains_USD)

            insample_data = df[columns]

            model = hmm.GaussianHMM(n_components=nstate, **param_dict, verbose=False)
            results = model.fit(insample_data)

            convergence = results.monitor_.converged
            all_states_found = np.isclose(a=(model.transmat_.sum(axis=1)), b=1).all()
            startprob_check = model.startprob_.sum() == 1
            good_model = convergence and all_states_found and startprob_check

            if good_model:
                try:
                    results_dict_df[stock].loc[nstate, "AIC"] = model.aic(insample_data)
                    results_dict_df[stock].loc[nstate, "BIC"] = model.bic(insample_data)
                except ValueError:
                    pass

            else:
                print(">" * 10, f"{stock} {nstate} did not converge")
                results_dict_df[stock].loc[nstate, "AIC"] = np.inf
                results_dict_df[stock].loc[nstate, "BIC"] = np.inf

    return results_dict_df


# In[46]:


results_dict_df_univ = fit_hmm_model(
    df, params["assetlist"], range_states, param_dict, contains_vol=False, contains_USD=False
)


# In[47]:


def select_best_model(
    df: pd.DataFrame,
    results_dict: dict,
    tickerlist: list,
    param_dict: dict,
    contains_vol: bool,
    contains_USD: bool,
):
    """"""
    aic_best_model = {stock: None for stock in tickerlist}
    bic_best_model = {stock: None for stock in tickerlist}

    for stock in tickerlist:
        columns = generate_columns(stock, contains_vol, contains_USD)
        insample_data = df[columns]

        best_aic_nstate = results_dict[stock]["AIC"].astype(float).idxmin()
        best_bic_nstate = results_dict[stock]["BIC"].astype(float).idxmin()

        print(
            f"For stock {stock}, best AIC: {best_aic_nstate} best BIC: {best_bic_nstate}"
        )

        aic_best_model[stock] = hmm.GaussianHMM(
            n_components=best_aic_nstate, **param_dict
        ).fit(insample_data)

        bic_best_model[stock] = hmm.GaussianHMM(
            n_components=best_bic_nstate, **param_dict
        ).fit(insample_data)

    return aic_best_model, bic_best_model


# In[48]:


aic_best_model_univ, bic_best_model_univ = select_best_model(
    df=df,
    results_dict=results_dict_df_univ,
    tickerlist=params["assetlist"],
    param_dict=param_dict,
    contains_vol=False,
    contains_USD=False,
)


# # Generating out of sample data

# In[49]:


name = f'finaldf_test_{params["tablename"]}.pickle'
filename = os.path.join(dataroute, name)
with open(filename, "rb") as handle:
    df_test = pickle.load(handle)


# In[50]:


def return_residuals(actual: pd.DataFrame, forecasts: pd.DataFrame):
    residuals = (actual - forecasts)
    return residuals


# In[51]:


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
    nstate = model.n_components
    columns = oos_data.columns

    split_date = oos_data.index[0]
    dates_to_forecast = len(oos_data.index)

    probabilities = pd.DataFrame(columns=range(nstate), index=oos_data.index)
    forecasts = pd.DataFrame(columns=oos_data.columns, index=oos_data.index)

    full_data = pd.concat([insample_data, oos_data])
    del insample_data

    # vamos a implementar recursive window forecasting

    index = full_data.index
    end_loc = np.where(index >= split_date)[0].min()
    # esto es un int del iloc
    # preciso usar ints de iloc porque el timedelta se me va a romper con el fin de semana
    rolling_window = 252

    nstate = model.n_components
    model = hmm.GaussianHMM(n_components=nstate, **param_dict, verbose=False)

    model_list = []
    counter = 0

    for i in range(1, dates_to_forecast):
        date_of_first_forecast = full_data.index[end_loc + i - 1]

        fitstart = end_loc - rolling_window + i
        fitend = end_loc + i

        # fit model with last year
        fit_data = full_data.iloc[fitstart:fitend][columns]
        res = model.fit(fit_data)
        model_list.append(res)

        # obtenemos las probabilidades por estado del ultimo dia
        # son las probabilidades que maximizan la log/likelihood de toda la secuencia
        index = len(model_list)
        while index > 0:
            try:
                add_count = False
                last_day_state_probs = res.predict_proba(fit_data)[-1]
                probabilities.loc[date_of_first_forecast] = last_day_state_probs
                index = 0

            except ValueError:
                # this happens when startprob_ must sum to 1 (got nan)
                # si el modelo falla en el predict_proba, se utiliza el de t-1
                add_count = True
                index = index - 1
                res = model_list[index]

                if not "last_day_state_probs" in locals():
                    # this checks for failure of estimation in the first day
                    last_day_state_probs = np.full(nstate, (1 / nstate))
                    # inputs a flat prior if it has no previous day to fall back on

        if add_count:
            counter = counter + 1
        # model.means_ es es la media condicional a cada estado
        # cada columna representa cada columna del dataset
        # cada fila es un estado
        # el producto punto entre este y las probabilidades del ultimo día me da la media esperada por cada columna
        expected_means = np.dot(last_day_state_probs, model.means_)
        forecasts.loc[date_of_first_forecast] = expected_means

    pct_nan = forecasts.iloc[:, 0].isna().sum() / len(forecasts.index) * 100

    if pct_nan > 5:
        warnings.warn(f"{oos_data.columns[0]} % na: {pct_nan}")

    forecasts.fillna(method="ffill", inplace=True)

    residuals = return_residuals(oos_data, forecasts)

    print("failed models: ", counter)
    return probabilities, forecasts, residuals, counter


# In[52]:


def generate_and_save_samples(
    best_model_dict: dict,
    modeltype: str,
    criterion: str,
    insample_data: pd.DataFrame,
    oos_data: pd.DataFrame,
    tickerlist: list,
    contains_vol: bool,
    contains_USD: bool,
):
    probabilities = {stock: None for stock in tickerlist}
    forecasts = {stock: None for stock in tickerlist}
    residuals = {stock: None for stock in tickerlist}
    failed = {stock: None for stock in tickerlist}

    print(">" * 10, modeltype, criterion)

    for stock in tickerlist:
        print(stock)
        columns = generate_columns(
            stock=stock, contains_vol=contains_vol, contains_USD=contains_USD
        )

        proba, fcast, resid, fails = generate_HMM_samples_residuals(
            best_model_dict[stock],
            insample_data=insample_data[columns],
            oos_data=oos_data[columns],
        )

        probabilities[stock] = proba
        forecasts[stock] = fcast
        residuals[stock] = resid
        failed[stock] = fails

    save_as_pickle(
        data=forecasts,
        resultsroute=params["resultsroute"],
        model_type=f"HMM_{modeltype}",
        tablename=params["tablename"],
        criterion=criterion,
        type_save="forecasts",
    )

    save_as_pickle(
        data=residuals,
        resultsroute=params["resultsroute"],
        model_type=f"HMM_{modeltype}",
        tablename=params["tablename"],
        criterion=criterion,
        type_save="residuals",
    )

    save_as_pickle(
        data=failed,
        resultsroute=params["resultsroute"],
        model_type=f"HMM_{modeltype}",
        tablename=params["tablename"],
        criterion=criterion,
        type_save="model_fails",
    )


# In[53]:


models_dict = {
    "aic": {
        "univ": (aic_best_model_univ, False, False)
    },
    "bic": {
        "univ": (bic_best_model_univ, False, False)
    },
}


# In[54]:


for criterion, type_dict in models_dict.items():
    for modeltype, tupla in type_dict.items():
        best_dict, contains_vol, contains_USD = tupla
        try:
            generate_and_save_samples(
                best_model_dict=best_dict,
                modeltype=modeltype,
                criterion=criterion,
                insample_data=df,
                oos_data=df_test,
                tickerlist=params["assetlist"],
                contains_vol=contains_vol,
                contains_USD=contains_USD,
            )
        except UnboundLocalError:
            print(f"MODEL FALILURE: {criterion}, {modeltype}")

