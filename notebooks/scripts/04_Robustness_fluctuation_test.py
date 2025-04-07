#!/usr/bin/env python
# coding: utf-8

# # Comparison
# 

# In[ ]:


import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from scipy.stats import norm

pd.set_option("display.max_columns", None)


# In[ ]:


from scripts.params import get_params
from scripts.aux_functions import get_all_results_matching, subset_of_columns, clean_modelname

params = get_params()


# In[ ]:


dataroute = params["dataroute"]
resultsroute = params["resultsroute"]
dumproute = params["dumproute"]
graphsroute = params["graphsroute"]
dmroute=params["dmroute"]
gwroute=params["gwroute"]


# In[ ]:


start_test = params["start_test"]
local_suffix = params["local_suffix"]


# In[ ]:


name = f'finaldf_test_{params["tablename"]}.pickle'
filename = os.path.join(dataroute, name)
with open(filename, "rb") as handle:
    df_test = pickle.load(handle)
    
df_test.index=pd.to_datetime(df_test.index.copy())


# In[ ]:


all_forecasts = get_all_results_matching(params["resultsroute"], ["best_forecast"])
all_residuals = get_all_results_matching(params["resultsroute"], ["best_residuals"])


# In[ ]:


def open_pickle_route(route:str):
    with open(route, "rb") as file:
        dictionary = pickle.load(file)
    return dictionary


# In[ ]:


def create_prefix(picklename:str):
    picklename=picklename.replace(f"""{params["tablename"]}_""", "").replace(".pickle", "").replace("_residuals", "").replace("_forecasts", "").replace("best", "")
    return picklename


# In[ ]:


def concat_dictionary(dictionary:dict, prefix:str):
    colname_list=[]
    df_list=[]
    for key, value in dictionary.items():
        value.index = pd.to_datetime(value.index)
        value = subset_of_columns(value, "log_rets", "USD")
        
        df_list.append(value)
        
        colname = prefix + key
        colname_list.append(colname)
    
    pickledf = pd.concat(df_list, axis=1, join="outer")
    pickledf.columns = colname_list
    
    return pickledf


# In[ ]:


def aggregate_single_pickle(picklename:str, pickleroute:str):
    prefix = create_prefix(picklename)
    dictionary = open_pickle_route(pickleroute)
    pickledf = concat_dictionary(dictionary, prefix)
    return pickledf


# In[ ]:


def aggregate_dict(dictionary:dict):
    pickledf_list=[]
    for picklename, pickleroute in dictionary.items():
        pickledf = aggregate_single_pickle(picklename, pickleroute)
        pickledf_list.append(pickledf)
    aggdf = pd.concat(pickledf_list, axis=1, join="outer")
    
    return aggdf


# In[ ]:


forecasts = aggregate_dict(all_forecasts)
residuals = aggregate_dict(all_residuals)


# In[ ]:


lower_date=pd.to_datetime(params["start_test"])+pd.Timedelta(days=1)
higher_date=pd.to_datetime(params["end_test"])-pd.Timedelta(days=1)

forecasts_df=forecasts[lower_date:higher_date].copy()
residual_df=residuals[lower_date:higher_date].copy()
df_test = df_test[lower_date:higher_date].copy()


# ## Separating in different stocks

# In[ ]:


def separate_by_stock(df:pd.DataFrame):
     stock_dict={}

     for stock in params["assetlist"]:
          if params["local_suffix"] in stock:
               stock_dict[stock]= subset_of_columns(residual_df, stock)
          else:
               stock_dict[stock]= subset_of_columns(residual_df, stock, params["local_suffix"])    
     
     return stock_dict      


# In[ ]:


forecasts_by_stock=separate_by_stock(forecasts_df)
residuals_by_stock=separate_by_stock(residual_df)


# In[ ]:


for df_clean, name in zip([forecasts_by_stock, residuals_by_stock], ["forecasts", "residuals"]):
    bystockname = name + "_by_stock_" + params["tablename"] + ".pickle"
    with open(os.path.join(resultsroute, bystockname), "wb") as handle:
        pickle.dump(df_clean, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:


def delete_in_column_names(df:pd.DataFrame, string:str):
    new_cols=[]
    for col in df.columns:
        col=col.replace(string, "")
        new_cols.append(col)
    df=df.set_axis(labels=new_cols, axis=1)
    return df


# # Fluctiation test

# In[ ]:


def fluctuation_test(real_values, forecast_1, forecast_2, window=60, loss="mse", significance_level=0.05):
    """
    Giacomini-Rossi Fluctuation Test for forecast comparison with symmetric critical value bands.

    Returns:
        pd.DataFrame with test_stat, upper_crit, lower_crit
    """
    assert len(real_values) == len(forecast_1) == len(forecast_2), "Input series must have equal length"

    # Compute loss differential
    if loss == "mse":
        d_t = (real_values - forecast_1)**2 - (real_values - forecast_2)**2
    elif loss == "mae":
        d_t = np.abs(real_values - forecast_1) - np.abs(real_values - forecast_2)
    else:
        raise ValueError("Unsupported loss function.")

    test_stats = []
    upper_crit = []
    lower_crit = []
    dates = []

    z = norm.ppf(1 - significance_level / 2)  # two-sided

    for i in range(window, len(d_t)):
        d_window = d_t.iloc[i - window:i]
        mean = d_window.mean()
        std = d_window.std(ddof=1)
        stat = np.sqrt(window) * mean / std

        test_stats.append(stat)
        upper_crit.append(z)
        lower_crit.append(-z)
        dates.append(d_t.index[i])

    return pd.DataFrame({
        "test_stat": test_stats,
        "upper_crit": upper_crit,
        "lower_crit": lower_crit
    }, index=dates)


# In[ ]:


def plot_fluctuation_test(result_df, title="", savefig=False, path="", filename="fluctuation_test"):
    plt.figure(figsize=(10, 5))
    plt.plot(result_df.index, result_df["test_stat"], label="Test Statistic", color="blue")
    plt.axhline(y=result_df["upper_crit"].iloc[0], linestyle="--", color="red", label="Upper Critical Value")
    plt.axhline(y=result_df["lower_crit"].iloc[0], linestyle="--", color="green", label="Lower Critical Value")
    plt.fill_between(result_df.index, result_df["lower_crit"], result_df["upper_crit"], color="gray", alpha=0.1, label="Non-rejection Region")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Fluctuation Test Statistic")
    plt.legend()
    plt.tight_layout()

    if savefig:
        plt.savefig(os.path.join(path, filename + ".png"), dpi=300)

    plt.show()


# In[ ]:


subset_to_test=["^BVSP", "BVSP_FX"]


# In[ ]:


for stock in subset_to_test:
    real_values = subset_of_columns(df_test, f"{stock}_log_rets").squeeze()
    forecasts = delete_in_column_names(forecasts_by_stock[stock].fillna(0), f"_{stock}")

    columns = forecasts.columns
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            model1, model2 = columns[i], columns[j]
            result = fluctuation_test(real_values, forecasts[model1], forecasts[model2], window=60, loss="mse")
            title = f"Fluctuation Test: {stock} {model1} vs {model2}"
            plot_fluctuation_test(result, title=title, savefig=True, path=gwroute, filename=f"{stock}_{model1}_vs_{model2}")

