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


dataroute = os.path.join("..", "data")
dumproute = os.path.join("..", "dump")
resultsroute = os.path.join("..", "results")


# In[5]:


from scripts.params import get_params

params = get_params()


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


tickerlist = params["tickerlist"]


# In[8]:


df.head(1)


# In[9]:


df_test.head(1)


# In[10]:


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


# In[11]:


selected_orders = VAR(df[["BBAR_log_rets", "BBAR_gk_vol"]]).select_order(
    maxlags=None, trend="c"
)
selected_orders.selected_orders


# In[12]:


def generate_VAR_samples_residuals(
    stock: str,
    lags: int,
    insample_data: pd.DataFrame,
    oos_data: pd.DataFrame,
    contains_vol: bool,
    contains_USD: bool,
):
    # pseudocodigo
    # agarra el mejor modelo (esto con una cantidad optima de params ya esta)
    # k = cantidad de params
    # fittear t-j con t-j-252d
    columns = generate_columns(
        stock=stock, contains_vol=contains_vol, contains_USD=contains_USD
    )

    split_date = oos_data.index[0]
    dates_to_forecast = len(oos_data.index)

    oos_data = pd.concat([insample_data[columns], oos_data[columns]])
    del insample_data

    index = oos_data.index
    end_loc = np.where(index >= split_date)[0].min()

    rolling_window = 252

    residuals = pd.DataFrame()

    for i in range(1, dates_to_forecast):
        fitstart = end_loc - rolling_window + i
        fitend = end_loc + i

        stock_data = oos_data.iloc[fitstart:fitend]

        model = VAR(stock_data)
        results = model.fit(lags)
        
        resid = results.resid.iloc[-1:]
        residuals = pd.concat([residuals, resid], axis=0)
        
    return residuals


# In[32]:


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

    residuals = generate_VAR_samples_residuals(
        stock=stock,
        lags=best_lag,
        insample_data=insample_data,
        oos_data=oos_data,
        contains_vol=contains_vol,
        contains_USD=contains_USD,
    )
    
    assert type(residuals)==pd.DataFrame
    
    return best_lag, residuals


# In[33]:


def save_as_pickle(data, contains_USD: bool, criterion: str, type_save: str):
    if contains_USD:
        string="multiv"
    else:
        string="with_vol"
    
    with open(
        os.path.join(
            resultsroute,
            f"""VAR_{string}_{params["tablename"]}_{criterion}_best_{type_save}.pickle""",
        ),
        "wb",
    ) as output_file:
        pickle.dump(data, output_file)


# In[42]:


best_lags = {
    "aic": {"contains_USD=True": {}, "contains_USD=False": {}},
    "bic": {"contains_USD=True": {}, "contains_USD=False": {}},
}

best_residuals = copy.deepcopy(best_lags)

for criterion in ["aic", "bic"]:
    for contains_USD in [True, False]:
        usdstring = f"contains_USD={contains_USD}"

        for stock in tickerlist:
            best_lag, residuals = estimate_best_residuals(
                stock=stock,
                criterion=criterion,
                insample_data=df,
                oos_data=df_test,
                contains_vol=True,
                contains_USD=contains_USD,
            )

            pct_nan = residuals.iloc[:, 0].isna().sum() / len(residuals.index) * 100

            if pct_nan > 5:
                warnings.warn(f"{stock} % na: {pct_nan}")

            residuals.fillna(method="ffill", inplace=True)

            best_lags[criterion][usdstring][stock] = best_lag
            
            best_residuals[criterion][usdstring][stock] = residuals

        save_as_pickle(
            data=best_lags[criterion][usdstring],
            contains_USD=contains_USD,
            criterion=criterion,
            type_save="lags",
        )
        save_as_pickle(
            data=best_residuals[criterion][usdstring],
            contains_USD=contains_USD,
            criterion=criterion,
            type_save="residuals",
        )

