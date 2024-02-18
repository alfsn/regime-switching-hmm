#!/usr/bin/env python
# coding: utf-8

# # Comparison
# 

# In[2]:


import pandas as pd
import numpy as np
import os

pd.set_option('display.max_columns', None)


# In[3]:


from scripts.params import get_params

params = get_params()


# In[4]:


dataroute = os.path.join("..", "data")
dumproute = os.path.join("..", "dump")
resultsroute = os.path.join("..", "results")


# In[5]:


all_residuals = {}

for filename in os.listdir(resultsroute):
    file_path = os.path.join(resultsroute, filename)
    if os.path.isfile(file_path) and "residual" in filename:
        all_residuals[filename] = file_path

print(all_residuals)


# In[6]:


residual_df = pd.DataFrame()

for name, dir in all_residuals.items():
    dict_with_dfs = pd.read_pickle(dir)
    print(name)
    for stock in dict_with_dfs.keys():
        if type(dict_with_dfs[stock]) == pd.Series:
            # univariate models are saved as series
            df = pd.DataFrame(dict_with_dfs[stock])
        else:
            try:
                # multivariate models are saved as dataframes
                df = pd.DataFrame(dict_with_dfs[stock][f"{stock}_log_rets"])
            except:  # TODO: SACAR ESTO! Es un chanchullo pq hay algunas que son guardadas como None
                pass

        modelname = (
            name.replace("residuals.pickle", "")
            .replace(params["tablename"], "")
            .replace("__", "_")
            .replace("__", "_")
        )
        df.columns = [modelname + "_" + stock]
        residual_df = pd.merge(
            residual_df, df, left_index=True, right_index=True, how="outer"
        )


# In[7]:


residual_df.tail(4)


# In[11]:


# estadisticos de nans
(residual_df.isna().sum() / len(residual_df.index) * 100).describe()


# In[28]:


# estadisticos de nans
((residual_df.isna().sum(axis=0) / len(residual_df.index)) * 100).nlargest(10)
# VAR tiene problemas con NANs


# In[38]:


# separo entre aic y bic
aic_residuals=pd.DataFrame(index=residual_df.index)
bic_residuals=pd.DataFrame(index=residual_df.index)

for criteria, df in {"aic":aic_residuals, "bic":bic_residuals}.items():
    for column in residual_df.columns:
        if criteria in column:
            df[column]=residual_df[column].copy()
            
bic_residuals.tail()


# In[56]:


model_list = ["GARCH", "HMM_univ", "HMM_multiv", "VAR_multiv", "VAR_univ"]

aggregating_dict = {"aic": {}, "bic": {}}

for criteria, dataframe in zip(("aic", "bic"), (aic_residuals, bic_residuals)):
    for model in model_list:
        filtered_columns = [col for col in dataframe.columns if model in col]
        aggregating_dict[criteria][model] = dataframe[filtered_columns].copy()

aggregating_dict["bic"]["GARCH"].head()


# In[83]:


metrics_df = pd.DataFrame(index=["mse", "meanabs", "medianabs"])

for criteria in aggregating_dict.keys():
    for model in aggregating_dict[criteria].keys():
        metrics_df.loc["mse", f"{criteria}_{model}"] = (
            ((aggregating_dict[criteria][model]) ** 2).mean().mean()
        )
        metrics_df.loc["meanabs", f"{criteria}_{model}"] = (
            ((aggregating_dict[criteria][model]).abs()).mean().mean()
        )
        metrics_df.loc["medianabs", f"{criteria}_{model}"] = (
            ((aggregating_dict[criteria][model]).abs()).median().median()
        )

metrics_df = metrics_df * 100
metrics_df


# In[87]:


for criteria in ["aic", "bic"]:
    print(criteria)
    filtered_columns = [col for col in metrics_df.columns if criteria in col]
    for metric in metrics_df.index:
        print(metric)
        print(metrics_df[filtered_columns].loc[metric].idxmax())
        print(np.round(metrics_df[filtered_columns].loc[metric].max(), 5))
        print()
    print()

