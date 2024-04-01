#!/usr/bin/env python
# coding: utf-8

# ## Startup

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm

import logging
import os
import pickle
import warnings


# In[2]:


logging.captureWarnings(True)
hmm.logging.disable(level=80)


# In[3]:


random_state = 42
np.random.seed(random_state)


# In[4]:


from scripts.params import get_params

params = get_params()


# ## Data Retrieval

# In[5]:


dataroute = os.path.join("..", "data")
dumproute = os.path.join("..", "dump")
resultsroute = os.path.join("..", "results")


# In[6]:


name = f'finaldf_train_{params["tablename"]}.pickle'
filename = os.path.join(dataroute, name)
with open(filename, "rb") as handle:
    df = pickle.load(handle)


# In[7]:


df.head()


# In[8]:


tickerlist = params["tickerlist"]


# ## HMM Training

# In[9]:


range_states = range(1, 16)
emptydf = pd.DataFrame(columns=["AIC", "BIC"], index=range_states)
emptydf.fillna(np.inf, inplace=True)
results_dict_df = {stock: emptydf for stock in tickerlist}


# In[10]:


param_dict = {
    "covariance_type": "diag",
    "n_iter": 500,
    "random_state": random_state,
    # no voy a usar startprob_prior por devlog 20-06-23
}


# In[11]:


def generate_columns(stock: str, contains_vol: bool, contains_USD: bool):
    """Devuelve una lista con los nombres de columnas para distintas especificaciones"""
    columns = []
    columns.append(f"{stock}_log_rets")

    if contains_vol:
        columns.append(f"{stock}_gk_vol")

    if contains_USD:
        columns.append(f"USD_log_rets")
        columns.append(f"USD_gk_vol")

    return columns


# In[12]:


def fit_hmm_model(
    df: pd.DataFrame,
    tickerlist: list,
    range_states,
    param_dict: dict,
    contains_vol: bool,
    contains_USD: bool,
):

    results_dict_df = {}

    for stock in tickerlist:
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


# In[13]:


results_dict_df_univ = fit_hmm_model(
    df, tickerlist, range_states, param_dict, contains_vol=False, contains_USD=False
)


# In[14]:


results_dict_df_with_vol = fit_hmm_model(
    df, tickerlist, range_states, param_dict, contains_vol=True, contains_USD=False
)


# In[15]:


results_dict_df_multi = fit_hmm_model(
    df, tickerlist, range_states, param_dict, contains_vol=True, contains_USD=True
)


# In[16]:


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


# In[17]:


aic_best_model_univ, bic_best_model_univ = select_best_model(
    df=df,
    results_dict=results_dict_df_univ,
    tickerlist=tickerlist,
    param_dict=param_dict,
    contains_vol=False,
    contains_USD=False,
)


# In[18]:


aic_best_model_with_vol, bic_best_model_with_vol = select_best_model(
    df=df,
    results_dict=results_dict_df_with_vol,
    tickerlist=tickerlist,
    param_dict=param_dict,
    contains_vol=False,
    contains_USD=False,
)


# In[19]:


aic_best_model_multi, bic_best_model_multi = select_best_model(
    df=df,
    results_dict=results_dict_df_multi,
    tickerlist=tickerlist,
    param_dict=param_dict,
    contains_vol=False,
    contains_USD=False,
)


# # Generating out of sample data

# In[20]:


name = f'finaldf_test_{params["tablename"]}.pickle'
filename = os.path.join(dataroute, name)
with open(filename, "rb") as handle:
    df_test = pickle.load(handle)


# In[21]:


def return_residuals(actual: pd.DataFrame, forecasts: pd.DataFrame):
    residuals = (actual - forecasts)
    return residuals


# In[27]:


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


# In[28]:


def save_as_pickle(data, modeltype: str, criterion: str, type_save: str):
    with open(
        os.path.join(
            resultsroute,
            f"""HMM_{modeltype}_{params["tablename"]}_{criterion}_best_{type_save}.pickle""",
        ),
        "wb",
    ) as output_file:
        pickle.dump(data, output_file)


# In[29]:


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
        data=residuals, modeltype=modeltype, criterion=criterion, type_save="residuals"
    )
    save_as_pickle(
        data=forecasts, modeltype=modeltype, criterion=criterion, type_save="forecasts"
    )
    save_as_pickle(
        data=failed, modeltype=modeltype, criterion=criterion, type_save="model_fails"
    )


# In[30]:


models_dict = {
    "aic": {
        "univ": (aic_best_model_univ, False, False),
        "with_vol": (aic_best_model_with_vol, True, False),
        "multiv": (aic_best_model_multi, True, True),
    },
    "bic": {
        "univ": (bic_best_model_univ, False, False),
        "with_vol": (bic_best_model_with_vol, True, False),
        "multiv": (bic_best_model_multi, True, True),
    },
}


# In[31]:


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
                tickerlist=params["tickerlist"],
                contains_vol=contains_vol,
                contains_USD=contains_USD,
            )
        except UnboundLocalError:
            print(f"MODEL FALILURE: {criterion}, {modeltype}")


# In[33]:


file=f"""HMM_multiv_{params["tablename"]}_aic_best_residuals.pickle"""
with open(os.path.join(resultsroute, file), "rb") as f:
    opened_pickle=pickle.load(f)


# In[35]:


opened_pickle[params["index"]].tail()


# # Graficando

# In[ ]:


def plot_close_rets_vol(model, data, key, IC):
    prediction = model.predict(data)
    states = set(prediction)

    fig = plt.figure(figsize=(20, 20))
    plt.tight_layout()
    plt.title(
        f"{key} Log returns and intraday Vol\n{model.n_components} states / best by {IC}"
    )

    for subplot, var in zip(range(1, 3), data.columns):
        plt.subplot(2, 1, subplot)
        for i in set(prediction):
            state = prediction == i
            x = data.index[state]
            y = data[var].iloc[state]
            plt.plot(x, y, ".")
        plt.legend(states, fontsize=16)

        plt.grid(True)
        plt.xlabel("datetime", fontsize=16)
        plt.ylabel(var, fontsize=16)

    plt.savefig(os.path.join(resultsroute, "graphs", f"HMM", f"{key}_model_{IC}.png"))


# In[ ]:


# for dictionary, IC in zip([aic_best_model, bic_best_model], ["AIC", "BIC"]):
#    for key, model in dictionary.items():
#        columns = [f"{stock}_log_rets", f"{stock}_gk_vol"]
#        insample_data = df[columns]
#        oos_data = df_test[columns]
#        train_end = insample_data.index.max()
#        data = pd.concat([insample_data, oos_data])
#
#        plot_close_rets_vol(model, data, key, IC)

