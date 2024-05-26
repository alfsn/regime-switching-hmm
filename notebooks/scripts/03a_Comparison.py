#!/usr/bin/env python
# coding: utf-8

# # Comparison
# 

# In[18]:


import pandas as pd
import numpy as np
import os
import pickle

pd.set_option("display.max_columns", None)


# In[19]:


from scripts.params import get_params
from scripts.aux_functions import get_all_results_matching, subset_of_columns, clean_modelname

params = get_params()


# In[20]:


from scripts.epftoolbox_dm_gw import DM, plot_multivariate_DM_test, GW, plot_multivariate_GW_test


# In[21]:


dataroute = params["dataroute"]
resultsroute = params["resultsroute"]
dumproute = params["dumproute"]
graphsroute = params["graphsroute"]
dmroute=params["dmroute"]
gwroute=params["gwroute"]


# In[22]:


start_test = params["start_test"]
local_suffix = params["local_suffix"]


# In[23]:


name = f'finaldf_test_{params["tablename"]}.pickle'
filename = os.path.join(dataroute, name)
with open(filename, "rb") as handle:
    df_test = pickle.load(handle)
    
df_test.index=pd.to_datetime(df_test.index.copy())


# In[24]:


all_residual = get_all_results_matching(params["resultsroute"], ["best_residual"])
all_residuals = get_all_results_matching(params["resultsroute"], ["best_residuals"])


# In[25]:


def open_pickle_route(route:str):
    with open(route, "rb") as file:
        dictionary = pickle.load(file)
    return dictionary


# In[26]:


def create_prefix(picklename:str):
    picklename=picklename.replace(f"""{params["tablename"]}_""", "").replace(".pickle", "").replace("_residuals", "").replace("_forecasts", "").replace("best", "")
    return picklename


# In[27]:


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


# In[28]:


def aggregate_single_pickle(picklename:str, pickleroute:str):
    prefix = create_prefix(picklename)
    dictionary = open_pickle_route(pickleroute)
    pickledf = concat_dictionary(dictionary, prefix)
    return pickledf


# In[29]:


def aggregate_dict(dictionary:dict):
    pickledf_list=[]
    for picklename, pickleroute in dictionary.items():
        pickledf = aggregate_single_pickle(picklename, pickleroute)
        pickledf_list.append(pickledf)
    aggdf = pd.concat(pickledf_list, axis=1, join="outer")
    
    return aggdf


# In[32]:


forecasts = aggregate_dict(all_forecasts)
residuals = aggregate_dict(all_residuals)


# In[36]:


lower_date=pd.to_datetime(params["start_test"])+pd.Timedelta(days=1)
higher_date=pd.to_datetime(params["end_test"])-pd.Timedelta(days=1)

forecasts_df=forecasts[lower_date:higher_date].copy()
residual_df=residuals[lower_date:higher_date].copy()
df_test = df_test[lower_date:higher_date].copy()


# In[37]:


# estadisticos de nans
nans_forecasts=((forecasts_df.isna().sum(axis=0) / len(forecasts_df.index)) * 100).nlargest(10)
assert nans_forecasts.iloc[0]<5, "Forecast NANs above 5%"
nans_forecasts


# In[38]:


# estadisticos de nans
nans_residual=((residual_df.isna().sum(axis=0) / len(residual_df.index)) * 100).nlargest(10)
assert nans_residual.iloc[0]<5, "Forecast NANs above 5%"
nans_residual


# ## Separating in different stocks

# In[52]:


params["assetlist"]


# In[53]:


def separate_by_stock(df:pd.DataFrame):
     stock_dict={}

     for stock in params["assetlist"]:
          if params["local_suffix"] in stock:
               stock_dict[stock]= subset_of_columns(residual_df, stock)
          else:
               stock_dict[stock]= subset_of_columns(residual_df, stock, params["local_suffix"])    
     
     return stock_dict      


# In[54]:


forecasts_by_stock=separate_by_stock(forecasts_df)
residuals_by_stock=separate_by_stock(residual_df)


# In[55]:


for df_clean, name in zip([forecasts_by_stock, residuals_by_stock], ["forecasts", "residuals"]):
    bystockname = name + "_by_stock_" + params["tablename"] + ".pickle"
    with open(os.path.join(resultsroute, bystockname), "wb") as handle:
        pickle.dump(df_clean, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[56]:


def delete_in_column_names(df:pd.DataFrame, string:str):
    new_cols=[]
    for col in df.columns:
        col=col.replace(string, "")
        new_cols.append(col)
    df=df.set_axis(labels=new_cols, axis=1)
    return df


# In[57]:


for stock in forecasts_by_stock.keys():
    print(stock)
    real_values=subset_of_columns(df_test, f"{stock}_log_rets")
    forecasts=delete_in_column_names(forecasts_by_stock[stock].fillna(0), f"_{stock}")   

    plot_multivariate_DM_test(real_price=real_values, 
                            forecasts=forecasts.fillna(0), 
                            title=f"DM test {stock}",
                            savefig=True,
                            path=dmroute)


# In[58]:


best_models_by_stock={stock:None for stock in residuals_by_stock.keys()}

for stock, dataframe in residuals_by_stock.items():
    dataframe = delete_in_column_names(dataframe, f"_{stock}")
    metrics_df = pd.DataFrame(index=["mse", "meanabs", "medianabs"])

    for column in dataframe.columns:
        single_model=pd.DataFrame(dataframe[column])
        
        metrics_df.loc["mse", column] = (
            (single_model**2).mean().mean()
        )
        metrics_df.loc["meanabs", column] = (
            single_model.abs().mean().mean()
        )
        metrics_df.loc["medianabs", column] = (
            (single_model.abs()).median().median()
        )
    metrics_df = metrics_df * 100
    metrics_df = subset_of_columns(metrics_df, substring="", exclude="USD")
    
    best_dict={}
    for criterion in metrics_df.index:
        best_dict[criterion] = metrics_df.iloc[metrics_df.index==criterion].idxmin(axis="columns").values[0]
        
    best_models_by_stock[stock]= (metrics_df, best_dict)


# In[59]:


print(params["tickerlist"][0])
best_models_by_stock[params["tickerlist"][0]][1]


# In[60]:


best_models_by_stock[params["tickerlist"][0]][0]


# In[61]:


best_models_by_stock[params["tickerlist"][0]][0].rank(axis=1)


# In[62]:


agg_df=(pd.DataFrame().reindex_like(best_models_by_stock[params["tickerlist"][0]][0]))

for asset in params["tickerlist"]:
    ranks = best_models_by_stock[asset][0].rank(axis=1)
    agg_df = agg_df.add(ranks, fill_value=0)
agg_df = agg_df/len(params["tickerlist"])
    
agg_df.rank(axis=1, method="average").astype(int)


# In[63]:


agg_df.to_csv(os.path.join(resultsroute, f"""aggregate_results_df_{params["tablename"]}.csv"""))


# In[64]:


criterion="mse"
print(f"Best overall performance by {criterion}")
agg_df.T.nsmallest(3, f"{criterion}").index.to_list()

