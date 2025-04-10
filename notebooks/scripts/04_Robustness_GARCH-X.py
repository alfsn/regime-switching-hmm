#!/usr/bin/env python
# coding: utf-8

# ## Startup

# In[1]:


import numpy as np
import pandas as pd

import arch

import os
import pickle
import warnings


# In[2]:


np.random.seed(42)


# In[3]:


from scripts.params import get_params
from scripts.aux_functions import save_as_pickle

params = get_params()


# In[4]:


dataroute = params["dataroute"]
resultsroute = params["resultsroute"]
dumproute = params["dumproute"]


# ## Data Retrieval

# In[5]:


name = f'finaldf_train_{params["tablename"]}.pickle'
filename = os.path.join(dataroute, name)
with open(filename, "rb") as handle:
    df = pickle.load(handle)


# ## GARCHX Training

# In[6]:


# Define the range of p and q values
alpha_values = [1, 2, 3, 4]  # estos son los valores de los lags in mean del AR.
p_values = [1, 2, 3]  # Example: p values
q_values = [0, 1, 2, 3]  # Example: q values
# all models with q=0 are exclusively ARCH (non-GARCH)


# In[7]:


models = {}
predict = {}


# In[8]:


best_aic = {}
best_bic = {}


# In[9]:


def check_best_aic(key, model, previous_best: float, p: int, q: int, dist: str):
    """
    AIC is better when lower.
    """
    if model == None:
        pass
    else:
        if model.aic < previous_best:
            best_aic[key] = {
                "model": model,
                "aic": model.aic,
                "p": p,
                "q": q,
                "dist": dist,
            }


# In[10]:


def check_best_bic(key, model, previous_best: float, p: int, q: int, dist: str):
    """
    BIC is better when lower.
    """
    if model == None:
        pass
    else:
        if model.aic < previous_best:
            best_bic[key] = {
                "model": model,
                "bic": model.bic,
                "p": p,
                "q": q,
                "dist": dist,
            }


# In[11]:


# Estimate ARMA-ARCHX and ARMA-GARCHX models for different p and q values
nonconverged_models = 0
ok_models = 0

for key in params["assetlist"]:
    returns = df[f"{key}_log_rets"]
    exog_rets=df["USD_log_rets"]

    models[key] = {}
    predict[key] = {}

    best_aic[key] = {"aic": np.inf}
    best_bic[key] = {"bic": np.inf}

    for p in p_values:
        for q in q_values:
            for dist in ["Normal", "StudentsT"]:
                model = arch.arch_model(
                    returns,
                    mean="AR",
                    lags=1,
                    vol="Garch",
                    p=p,
                    q=q,
                    dist=dist,
                    x=exog_rets,
                    rescale=False,
                )
                results = model.fit(
                    options={"maxiter": 2000}, disp="off", show_warning=False
                )

                if results.convergence_flag != 0:
                    # 0 is converged successfully
                    # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_slsqp.html
                    results = None
                    nonconverged_models += 1
                else:
                    ok_models += 1

                check_best_aic(
                    key=key,
                    model=results,
                    previous_best=best_aic[key]["aic"],                    
                    p=p,
                    q=q,
                    dist=dist,
                )
                check_best_bic(
                    key=key,
                    model=results,
                    previous_best=best_bic[key]["bic"],                    
                    p=p,
                    q=q,
                    dist=dist,
                )

                models[key][(p, q, dist)] = results

print()
print(f"ok: {ok_models}")
print(f"nonconverged: {nonconverged_models}")


# # Residuals

# In[12]:


name = f'finaldf_test_{params["tablename"]}.pickle'
filename = os.path.join(dataroute, name)
with open(filename, "rb") as handle:
    df_test = pickle.load(handle)


# In[ ]:


def generate_GARCH_samples_residuals(
    model_dict: dict, insample_data: pd.DataFrame, oos_data: pd.DataFrame
):
    """
    Esta función come archmodelresults (que vienen del diccionario best_aic y best_bic),
    y hace pronósticos rolling (con ventana de 1 año (252 días habiles)),
    lo que devuelve samples y residuos.
    El método de pronóstico es de simulación

    Args:
        model_dict (_type_): _description_
        pd (_type_): _description_

    Returns:
        _type_: _description_
    """
    split_date = insample_data.index[-1]
    dates_to_forecast = len(oos_data.index)

    full_data = pd.concat([insample_data, oos_data])
    del insample_data

    # vamos a implementar recursive window forecasting
    # https://arch.readthedocs.io/en/latest/univariate/forecasting.html
    # https://arch.readthedocs.io/en/latest/univariate/univariate_volatility_forecasting.html#Recursive-Forecast-Generation

    index = full_data.index
    end_loc = np.where(index >= split_date)[0].min()
    # esto es un int del iloc
    # preciso usar ints de iloc porque el timedelta se me va a romper con el fin de semana
    rolling_window = 252

    forecasts = {}

    model = arch.arch_model(
        y=full_data.iloc[:, 0],
        mean="AR",
        lags=1,
        vol="Garch",
        p=model_dict["p"],
        q=model_dict["q"],
        dist=model_dict["dist"],
        x=full_data.iloc[:, 1], 
        rescale=False,
    )

    for i in range(0, dates_to_forecast):
        date_of_first_forecast = full_data.index[end_loc + i]

        res = model.fit(
            first_obs=end_loc - rolling_window + i, last_obs=end_loc + i, disp="off"
        )

        future_x = full_data.iloc[end_loc + i -1 : end_loc + i, 1:2]

        forecast = res.forecast(
            horizon=1,
            start=date_of_first_forecast,
            method="simulation",
            x=future_x
        ).mean.iloc[0]

        forecasts[forecast.name] = forecast

    forecasts = pd.DataFrame(forecasts).T
    forecasts.columns = pd.DataFrame(full_data.iloc[:, 0]).columns

    pct_nan = forecasts.iloc[:, 0].isna().sum() / len(forecasts.index) * 100

    if pct_nan > 5:
        warnings.warn(f"{full_data.columns[0]} % na: {pct_nan}")

    forecasts.fillna(method="ffill", inplace=True)

    residuals = oos_data - forecasts

    return forecasts, residuals


# In[14]:


forecasts_dict={"aic":{}, "bic":{}}
residuals_dict={"aic":{}, "bic":{}}

for criterion, dictionary in zip(["aic", "bic"], [best_aic, best_bic]):
    for stock in dictionary.keys():
        columns=[f"{stock}_log_rets", "USD_log_rets"]
        forecasts, residuals = generate_GARCH_samples_residuals(
            model_dict=dictionary[stock],
            insample_data=pd.DataFrame(df[columns]),
            oos_data=pd.DataFrame(df_test[columns])
            )

        forecasts_dict[criterion][stock]=forecasts
        residuals_dict[criterion][stock]=residuals     


# In[ ]:





# In[ ]:


for criterion, bestmodels in zip(["aic", "bic"], [best_aic, best_bic]):
    save_as_pickle(
        data=forecasts_dict[criterion],
        resultsroute=params["resultsroute"],
        model_type="GARCH-X",
        tablename=params["tablename"],
        criterion=criterion,
        type_save="forecasts",
    )

    save_as_pickle(
        data=residuals_dict[criterion],
        resultsroute=params["resultsroute"],
        model_type="GARCH-X",
        tablename=params["tablename"],
        criterion=criterion,
        type_save="residuals"
    )

    save_as_pickle(
        data=bestmodels,
        resultsroute=params["resultsroute"],
        model_type="GARCH-X",
        tablename=params["tablename"],
        criterion=criterion,
        type_save="models"
    )

