#!/usr/bin/env python
# coding: utf-8

# # Intermarket Comparison
# 

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import os
import pickle

pd.set_option("display.max_columns", None)


# In[2]:


from scripts.params import get_params
from scripts.aux_functions import subset_of_columns, clean_modelname, plot_residuals

params = get_params()


# In[3]:


datasets={}
for tablename in ["AR_^MERV", "BR_^BVSP", "MX_^MXX"]:
    filename = os.path.join("..", "data", tablename, f'finaldf_test_{tablename}.pickle')
    with open(filename, "rb") as handle:
        df_test = pickle.load(handle)
        
    df_test.index=pd.to_datetime(df_test.index.copy())
    datasets[tablename] = df_test
    del df_test


# In[4]:


forecasts={}
residuals={}
for tablename in ["AR_^MERV", "BR_^BVSP", "MX_^MXX"]:
    for dictionary, picklename in [(forecasts, "forecasts_by_stock_"), (residuals, "residuals_by_stock_")]:
        filename = os.path.join("..", "results", tablename, f'{picklename}{tablename}.pickle')
        with open(filename, "rb") as handle:
            by_stock = pickle.load(handle)
        dictionary[tablename] = by_stock
        del by_stock


# In[10]:


for col in ["GARCH", "HMM", "VAR"]:
    for tablename, asset in [("AR_^MERV", "^MERV"), ("BR_^BVSP","^BVSP"), ("MX_^MXX", "^MXX")]:
        
        resdf = subset_of_columns(residuals[tablename][asset], col)
        fig = plot_residuals(resdf, tablename, show=False, return_fig=True)


# In[20]:


def plot_residuals(df, stock, show=True, return_fig=False, ax=None):
  """
  Plots the residuals for a given stock on a specified axis.

  Args:
      df (pandas.DataFrame): The dataframe containing the data.
      stock (str): The name of the stock.
      show (bool, optional): Whether to display the plot directly. Defaults to True.
      return_fig (bool, optional): Whether to return the figure object. Defaults to False.
      ax (matplotlib.axes._axes.Axes, optional): The axis to plot on. Defaults to None.

  Returns:
      plt.Figure (optional): The figure object containing the plot (if return_fig is True).
  """

  if ax is None:  # Create figure and axis if ax is not provided
    fig, ax = plt.subplots(figsize=(12, 6))
  else:
    fig = None  # Avoid creating a new figure if ax is provided

  sb.lineplot(data=df, markers=True, ax=ax)  # Plot residuals on the axis

  # Add title, labels, and formatting
  plt.title(f"Residuals - {stock}", fontsize=14)
  plt.xlabel("Date", fontsize=12)
  plt.ylabel("Values", fontsize=12)
  plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
  plt.grid(True, linestyle="--", linewidth=0.5)
  plt.axhline(y=0, color="black", linestyle="--", linewidth=1.5, label="Zero Error")

  handles, labels = ax.get_legend_handles_labels()
  ax.legend(handles, labels, title="Series", loc="upper left", bbox_to_anchor=(1, 1))

  plt.tight_layout()

  # Handle show and return options
  if show:
    plt.show()  # Display the plot
  if return_fig:
    return fig  # Return the figure object (if created)


# In[21]:


def create_residual_grid(residuals):
  """
  Creates a 3x3 grid of residual plots for different models and assets.

  Args:
      residuals (dict): A dictionary containing residual dataframes for different models and assets.

  Returns:
      plt.Figure: The figure object containing the 3x3 grid of plots.
  """

  fig, axes = plt.subplots(3, 3, figsize=(15, 15))  # Create 3x3 grid figure

  # Iterate through models and assets
  row = 0
  col = 0
  for col_name in ["GARCH", "HMM", "VAR"]:
    for tablename, asset in [("AR_^MERV", "^MERV"), ("BR_^BVSP","^BVSP"), ("MX_^MXX", "^MXX")]:
      resdf = subset_of_columns(residuals[tablename][asset], col_name)
      plot_residuals(resdf, tablename, show=False, return_fig=fig, ax=axes[row, col])
      col += 1
      if col == 3:
        col = 0
        row += 1

  # Adjust layout and add a common title (optional)
  fig.suptitle("Residual Plots - Different Models and Assets", fontsize=16)
  plt.tight_layout()
  return fig


# In[22]:


create_residual_grid(residuals)


# In[ ]:





# In[ ]:





# In[80]:





# In[81]:


forecasts_df = create_df_from_results_dict(all_forecasts, "forecasts")


# In[82]:


forecasts_df = pd.concat([forecasts_df, subset_of_columns(df_test, "log_rets")])


# In[83]:


residual_df = create_df_from_results_dict(all_residuals, "residuals")


# In[84]:


lower_date=pd.to_datetime(params["start_test"])+pd.Timedelta(days=1)
higher_date=pd.to_datetime(params["end_test"])-pd.Timedelta(days=1)
residual_df=residual_df[lower_date:higher_date].copy()
df_test = df_test[lower_date:higher_date].copy()
residual_df.head()


# In[85]:


# estadisticos de nans
((residual_df.isna().sum(axis=0) / len(residual_df.index)) * 100).nlargest(10)


# In[86]:


# estadisticos de nans
((forecasts_df.isna().sum(axis=0) / len(forecasts_df.index)) * 100).nlargest(10)


# ## Separating in different stocks

# In[87]:


def separate_by_stock(df:pd.DataFrame):
     stock_dict={}

     for stock in params["tickerlist"]:
          if params["local_suffix"] in stock:
               stock_dict[stock]= subset_of_columns(residual_df, stock)
          else:
               stock_dict[stock]= subset_of_columns(residual_df, stock, params["local_suffix"])    
     
     return stock_dict      


# In[88]:


forecasts_by_stock=separate_by_stock(forecasts_df)
residuals_by_stock=separate_by_stock(residual_df)


# In[89]:


for df_clean, name in zip([forecasts_by_stock, residuals_by_stock], ["forecasts", "residuals"]):
    bystockname = name + "_by_stock_" + params["tablename"] + ".pickle"
    with open(os.path.join(resultsroute, bystockname), "wb") as handle:
        pickle.dump(df_clean, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[90]:


def delete_in_column_names(df:pd.DataFrame, string:str):
    new_cols=[]
    for col in df.columns:
        col=col.replace(string, "")
        new_cols.append(col)
    df=df.set_axis(labels=new_cols, axis=1)
    return df


# In[91]:


for stock in forecasts_by_stock.keys():
    print(stock)
    real_values=subset_of_columns(df_test, f"{stock}_log_rets")
    forecasts=delete_in_column_names(forecasts_by_stock[stock].fillna(0), f"_{stock}")   

    plot_multivariate_DM_test(real_price=real_values, 
                            forecasts=forecasts.fillna(0), 
                            title=f"DM test {stock}",
                            savefig=True,
                            path=dmroute)


# In[92]:


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


# In[93]:


print(params["tickerlist"][0])
best_models_by_stock[params["tickerlist"][0]][1]


# In[94]:


best_models_by_stock[params["tickerlist"][0]][0]


# In[95]:


best_models_by_stock[params["tickerlist"][0]][0].rank(axis=1)


# In[96]:


agg_df=(pd.DataFrame().reindex_like(best_models_by_stock[params["tickerlist"][0]][0]))

for asset in params["tickerlist"]:
    ranks = best_models_by_stock[asset][0].rank(axis=1)
    agg_df = agg_df.add(ranks, fill_value=0)
agg_df = agg_df/len(params["tickerlist"])
    
agg_df=agg_df.rank(axis=1, method="average").astype(int)
agg_df


# In[97]:


agg_df.to_csv(os.path.join(resultsroute, f"""aggregate_results_df_{params["tablename"]}.csv"""))


# In[98]:


agg_df.to_clipboard()


# In[99]:


criterion="mse"
print(f"Best overall performance by {criterion}")
agg_df.T.nsmallest(3, f"{criterion}").index.to_list()

