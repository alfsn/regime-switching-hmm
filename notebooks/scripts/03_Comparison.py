#!/usr/bin/env python
# coding: utf-8

# # Comparison
# 

# In[26]:


import pandas as pd
import numpy as np
import os

pd.set_option("display.max_columns", None)


# In[27]:


from scripts.params import get_params

params = get_params()


# In[28]:


dataroute = os.path.join("..", "data")
dumproute = os.path.join("..", "dump")
resultsroute = os.path.join("..", "results")


# In[46]:


start_test = params["start_test"]


# In[29]:


all_residuals = {}

for filename in os.listdir(resultsroute):
    file_path = os.path.join(resultsroute, filename)
    if os.path.isfile(file_path) and "residual" in filename:
        all_residuals[filename] = file_path

print(all_residuals)


# In[30]:


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


# In[48]:


residual_df = pd.DataFrame()

for name, dir in all_residuals.items():
    dict_with_dfs = pd.read_pickle(dir)
    print(name)

    for stock in dict_with_dfs.keys():
        df = get_only_log_rets(dict_with_dfs, stock)

        modelname = (
            name.replace("residuals.pickle", "")
            .replace("best", "")
            .replace(params["tablename"], "")
            .replace("__", "_")
            .replace("__", "_")
        )

        df.columns = [modelname + "_" + stock]

        residual_df = pd.merge(
            residual_df, df, left_index=True, right_index=True, how="outer"
        )

residual_df.index = pd.to_datetime(residual_df.index)
residual_df = residual_df[residual_df.index > start_test]


# In[32]:


def subset_of_columns(df: pd.DataFrame, substring: str):
    filtered_columns = [col for col in df.columns if substring in col]
    return df[filtered_columns]


# In[58]:


aic_residuals = subset_of_columns(residual_df, "aic")
bic_residuals = subset_of_columns(residual_df, "bic")


# In[52]:


# estadisticos de nans
(residual_df.isna().sum() / len(residual_df.index) * 100).describe()


# In[53]:


# estadisticos de nans
((residual_df.isna().sum(axis=0) / len(residual_df.index)) * 100).nlargest(10)
# VAR tiene problemas con NANs


# In[59]:


model_list = ["GARCH", "HMM_univ", "HMM_multiv", "VAR_multiv", "VAR_with_vol"]

aggregating_dict = {"aic": {}, "bic": {}}

for criteria, dataframe in zip(("aic", "bic"), (aic_residuals, bic_residuals)):
    for model in model_list:
        aggregating_dict[criteria][model] = subset_of_columns(dataframe, model)

aggregating_dict["bic"]["GARCH"].head()


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

