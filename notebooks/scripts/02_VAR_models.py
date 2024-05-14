#!/usr/bin/env python
# coding: utf-8

# ## Startup

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import aic, bic

import copy
import os
import pickle


# In[2]:


import warnings

warnings.filterwarnings("ignore")


# In[3]:


np.random.seed(42)


# In[4]:


from scripts.params import get_params
from scripts.aux_functions import generate_columns, save_as_pickle

params = get_params()


# In[5]:


dataroute = params["dataroute"]
resultsroute = params["resultsroute"]
dumproute = params["dumproute"]


# ## Data Retrieval

# In[6]:


name = f"""finaldf_train_{params["tablename"]}.pickle"""
filename = os.path.join(dataroute, name)
with open(filename, "rb") as handle:
    df = pickle.load(handle)

name = f'finaldf_test_{params["tablename"]}.pickle'
filename = os.path.join(dataroute, name)
with open(filename, "rb") as handle:
    df_test = pickle.load(handle)


# In[7]:


def generate_VAR_samples_residuals(
    stock: str,
    lags: int,
    insample_data: pd.DataFrame,
    oos_data: pd.DataFrame,
    contains_vol: bool,
    contains_USD: bool,
):
    columns = generate_columns(
        stock=stock, contains_vol=contains_vol, contains_USD=contains_USD
    )

    combined_data = pd.concat([insample_data[columns], oos_data[columns]])

    split_date = oos_data.index[0]
    dates_to_forecast = len(oos_data)

    fcast_holder = []
    resid_holder = []

    for i in range(0, dates_to_forecast):
        end_loc = combined_data.index.get_loc(split_date) + i
        fitstart = end_loc - 252
        fitend = end_loc

        stock_data = combined_data.iloc[fitstart:fitend]

        model = VAR(stock_data)
        results = model.fit(lags)

        fcast = results.forecast(y=stock_data.values, steps=1)
        resid = results.resid.iloc[-1:]
        
        fcast_holder.append(fcast)
        resid_holder.append(resid)

    forecasts = pd.DataFrame(np.concatenate(fcast_holder), columns=columns, index=oos_data.index)
    residuals = pd.DataFrame(np.concatenate(resid_holder), columns=columns, index=oos_data.index)

    return forecasts, residuals


# In[8]:


def estimate_best_residuals(
    stock: str,
    criterion: str,
    insample_data: pd.DataFrame,
    oos_data: pd.DataFrame,
    contains_vol: bool,
    contains_USD: bool,
):
    columns = generate_columns(
        stock=stock, contains_vol=contains_vol, contains_USD=contains_USD
    )

    selected_orders = VAR(insample_data[columns]).select_order(maxlags=15, trend="c")
    best_lag = selected_orders.selected_orders[criterion]

    forecasts, residuals = generate_VAR_samples_residuals(
        stock=stock,
        lags=best_lag,
        insample_data=insample_data,
        oos_data=oos_data,
        contains_vol=contains_vol,
        contains_USD=contains_USD,
    )

    assert type(residuals) == pd.DataFrame

    return best_lag, forecasts, residuals


# In[9]:


best_lags = {
    "aic": {"contains_USD=True": {}, "contains_USD=False": {}},
    "bic": {"contains_USD=True": {}, "contains_USD=False": {}},
}
best_forecasts = copy.deepcopy(best_lags)
best_residuals = copy.deepcopy(best_lags)

for criterion in ["aic", "bic"]:
    for contains_USD in [True, False]:
        usdstring = f"contains_USD={contains_USD}"

        for stock in params["assetlist"]:
            best_lag, forecasts, residuals = estimate_best_residuals(
                stock=stock,
                criterion=criterion,
                insample_data=df,
                oos_data=df_test,
                contains_vol=True,
                contains_USD=contains_USD,
            )

            pct_nan = forecasts.iloc[:, 0].isna().sum() / len(forecasts.index) * 100

            if pct_nan > 5:
                warnings.warn(f"{stock} % na: {pct_nan}")

            forecasts.fillna(method="ffill", inplace=True)
            residuals.fillna(method="ffill", inplace=True)

            best_lags[criterion][usdstring][stock] = best_lag
            best_forecasts[criterion][usdstring][stock] = forecasts
            best_residuals[criterion][usdstring][stock] = residuals

        if contains_USD:
            string = "multiv"
        else:
            string = "with_vol"

        save_as_pickle(
            data=best_lags[criterion][usdstring],
            resultsroute=params["resultsroute"],
            model_type=f"VAR_{string}",
            tablename=params["tablename"],
            criterion=criterion,
            type_save="lags",
        )
        
        save_as_pickle(
            data=best_forecasts[criterion][usdstring],
            resultsroute=params["resultsroute"],
            model_type=f"VAR_{string}",
            tablename=params["tablename"],
            criterion=criterion,
            type_save="forecasts",
        )

        save_as_pickle(
            data=best_residuals[criterion][usdstring],
            resultsroute=params["resultsroute"],
            model_type=f"VAR_{string}",
            tablename=params["tablename"],
            criterion=criterion,
            type_save="residuals",
        )


# In[10]:


for crit, d in best_residuals.items():
    for cols, values in d.items():
        for stock, dataframe in values.items():
            isna= dataframe.iloc[:,0].isna().sum()/len(dataframe.index)
            if isna>0:
                print(crit, stock, cols, isna)

