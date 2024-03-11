#!/usr/bin/env python
# coding: utf-8

# # Comparison
# 

# In[1]:


import pandas as pd
import numpy as np
import os
import pickle

pd.set_option("display.max_columns", None)


# In[2]:


from scripts.params import get_params

params = get_params()


# In[3]:


from scripts.epftoolbox_dm_gw import DM, plot_multivariate_DM_test, GW, plot_multivariate_GW_test


# In[4]:


dataroute = os.path.join("..", "data")
dumproute = os.path.join("..", "dump")
resultsroute = os.path.join("..", "results")
graphsroute = os.path.join(resultsroute, "graphs")


# In[5]:


start_test = params["start_test"]
local_suffix = params["local_suffix"]


# In[6]:


name = f'finaldf_test_{params["tablename"]}.pickle'
filename = os.path.join(dataroute, name)
with open(filename, "rb") as handle:
    df_test = pickle.load(handle)


# In[7]:


def get_all_results_matching(substring:str):
    all_results = {}

    for filename in os.listdir(resultsroute):
        file_path = os.path.join(resultsroute, filename)
        if os.path.isfile(file_path) and substring in filename:
            all_results[filename] = file_path

    print(all_results)
    return all_results


# In[8]:


all_forecasts = get_all_results_matching("forecast")
all_residuals = get_all_results_matching("residual")


# In[9]:


def get_only_log_rets(dict_with_dfs: dict, stock: str):
    if type(dict_with_dfs[stock]) == pd.Series:
        # univariate models are saved as series
        df = pd.DataFrame(dict_with_dfs[stock])

    else:
        try:
            # multivariate models are saved as dataframes
            df = pd.DataFrame(dict_with_dfs[stock][f"{stock}_log_rets"])
        except:  # TODO: SACAR ESTO! Es un chanchullo pq hay algunas que son guardadas como None
            pass
    return df


# In[10]:


def create_df_from_results_dict(results_dict:dict, substring_to_replace:str):
    created_df = pd.DataFrame()

    for name, dir in results_dict.items():
        dict_with_dfs = pd.read_pickle(dir)
        print(name)

        for stock in dict_with_dfs.keys():
            df = get_only_log_rets(dict_with_dfs, stock)

            modelname = (
                name.replace(f"{substring_to_replace}.pickle", "")
                .replace("best", "")
                .replace(params["tablename"], "")
                .replace("__", "_")
                .replace("__", "_")
            )

            df.columns = [modelname + "_" + stock]

            created_df = pd.merge(
                created_df, df, left_index=True, right_index=True, how="outer"
            )

    created_df.index = pd.to_datetime(created_df.index)
    created_df = created_df[created_df.index > start_test]
    return created_df


# In[11]:


forecasts_df = create_df_from_results_dict(all_forecasts, "forecasts")
forecasts_df.head(2)


# In[12]:


residual_df = create_df_from_results_dict(all_residuals, "residuals")
residual_df.head(2)


# In[13]:


# estadisticos de nans
((residual_df.isna().sum(axis=0) / len(residual_df.index)) * 100).nlargest(10)
# VAR tiene problemas con NANs


# ## Separating in different stocks

# In[14]:


def subset_of_columns(df: pd.DataFrame, substring: str, exclude:str=None):
    filtered_columns = [col for col in df.columns if substring in col] 
    
    if exclude is not None:
        filtered_columns = [col for col in filtered_columns.copy() if exclude not in col] 

    return df[filtered_columns]


# In[15]:


def separate_by_stock(df:pd.DataFrame):
     stock_dict={}

     for stock in params["tickerlist"]:
          if params["local_suffix"] in stock:
               stock_dict[stock]= subset_of_columns(residual_df, stock)
          else:
               stock_dict[stock]= subset_of_columns(residual_df, stock, params["local_suffix"])    
     
     return stock_dict      


# In[16]:


forecasts_by_stock=separate_by_stock(forecasts_df)
residuals_by_stock=separate_by_stock(residual_df)


# In[17]:


def delete_in_column_names(df:pd.DataFrame, string:str):
    new_cols=[]
    for col in df.columns:
        col=col.replace(string, "")
        new_cols.append(col)
    df=df.set_axis(labels=new_cols, axis=1)
    return df


# In[23]:


dmroute=os.path.join(graphsroute, "DM")
gwroute=os.path.join(graphsroute, "GW")

for stock in forecasts_by_stock.keys():
    print(stock)
    real_values=subset_of_columns(df_test, f"{stock}_log_rets")
    forecasts=delete_in_column_names(forecasts_by_stock[stock].fillna(0), f"__{stock}")   

    plot_multivariate_DM_test(real_price=real_values, 
                            forecasts=forecasts.fillna(0), 
                            title=f"DM test {stock}",
                            savefig=True,
                            path=dmroute)


# In[29]:


residuals_by_stock


# In[63]:


metrics_df = pd.DataFrame(index=["mse", "meanabs", "medianabs"])

for criteria, dictionary in aggregating_dict.items():
    for model, dataframe in dictionary.items():
        metrics_df.loc["mse", f"{criteria}_{model}"] = (
            (dataframe**2).mean().mean()
        )
        metrics_df.loc["meanabs", f"{criteria}_{model}"] = (
            dataframe.abs().mean().mean()
        )
        metrics_df.loc["medianabs", f"{criteria}_{model}"] = (
            (dataframe.abs()).median().median()
        )

metrics_df = metrics_df * 100
metrics_df


# In[64]:


for criteria in ["aic", "bic"]:
    print(criteria)
    filtered_columns = [col for col in metrics_df.columns if criteria in col]
    for metric in metrics_df.index:
        print(metric)
        print(metrics_df[filtered_columns].loc[metric].idxmin())
        print(np.round(metrics_df[filtered_columns].loc[metric].min(), 5))
        print()
    print()


# In[ ]:


DM(p_real=0,)


# In[ ]:




