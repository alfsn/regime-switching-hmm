#!/usr/bin/env python
# coding: utf-8

# ## Startup

# In[28]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, coint
from statsmodels.tools.eval_measures import rmse, aic

import os
import pickle


# In[2]:


np.random.seed(42)


# In[3]:


dataroute=os.path.join("..",  "data")
resultsroute=os.path.join("..",  "results")


# ## Data Retrieval

# In[6]:


start='2013-01-01'
end="2023-06-01"

name=f'processed_dataset_{start}_{end}.pickle'
filename=os.path.join(dataroute, name)
with open(filename, 'rb') as handle:
    data=pickle.load(handle)
    
name=f'finaldf_dataset_{start}_{end}.pickle'
filename=os.path.join(dataroute, name)
with open(filename, 'rb') as handle:
    df=pickle.load(handle)


# In[7]:


df.head(3)


# ## Test de Causalidad de Granger
# 
# **Null Hypothesis (H0)**: past values of one time series do not have any predictive power or influence on the other time series. Essentially, it says that knowing the past values of the first time series doesn't help us predict the future values of the second time series.
# 
# **Alternative Hypothesis (Ha)**: past values of one time series do have predictive power and influence over the other time series. If the alternative hypothesis is supported, it indicates that knowing the past values of the first time series can help us make better predictions about the future values of the second time series.

# In[14]:


def granger_causality_test(series1, series2, max_lag):
    result = grangercausalitytests(np.column_stack([series1, series2]), max_lag, verbose=False)
    return result[max_lag][0]['ssr_ftest'][1]


# In[15]:


def granger_causality_matrix(dataframe, max_lag=5):
    """
    Esta función estudia 
        cada FILA como potencial causa
        cada COLUMNA como potencial efecto
    """
    num_cols = len(dataframe.columns)
    granger_matrix = np.zeros((num_cols, num_cols))

    for i in range(num_cols):
        for j in range(num_cols):
            if i != j:
                p_value = granger_causality_test(dataframe.iloc[:, i], dataframe.iloc[:, j], max_lag=max_lag)
                granger_matrix[i, j] = p_value

    granger_df = pd.DataFrame(granger_matrix, columns=dataframe.columns, index=dataframe.columns)
    return granger_df


# In[24]:


causality_matrix=granger_causality_matrix(df, max_lag=12)
causality_matrix.head()


# In[25]:


causality_matrix[g_matrix<0.05]


# In[20]:


# TODO: interpretar resultados


# ## Test de Cointegración
# Because the matrix of results is not positive semidefinite (daily returns are sometimes negative), we must choose the Engle-Granger specification of the test.

# In[32]:


def engle_granger_cointegration_test(dataframe):
    pairs = []
    p_values = []

    # Iterate through all pairs of columns
    for i in range(len(dataframe.columns)):
        for j in range(i + 1, len(dataframe.columns)):
            series1 = dataframe.iloc[:, i]
            series2 = dataframe.iloc[:, j]
            
            # Perform the Engle-Granger cointegration test
            _, p_value, _ = coint(series1, series2)
            
            # Store the pair of variables and the p-value
            pairs.append((dataframe.columns[i], dataframe.columns[j]))
            p_values.append(p_value)

    # Create a DataFrame to store the results
    results_df = pd.DataFrame({'Pair': pairs, 'p_value': p_values})

    return results_df


# In[33]:


coint_matrix=engle_granger_cointegration_test(df)
coint_matrix.head()


# In[44]:


coint_matrix.p_value.plot(kind="hist")


# In[67]:


(coint_matrix.p_value<0.001).sum()/coint_matrix["p_value"].count()


# La totalidad de los p-valores están por debajo del 0.1%. Todo está cointegrado con todo.

# ## Test de estacionaridad

# In[53]:


def adf_test(dataframe):
    """
    Perform the Augmented Dickey-Fuller (ADF) test for multiple data series and return p-values.

    Parameters:
    - dataframe (pd.DataFrame): The input DataFrame with multiple time series columns.
    - significance_level (float): The significance level for the ADF test (default is 0.05).

    Returns:
    - pd.DataFrame: A DataFrame containing p-values for each time series.
    """

    p_values = []

    for column in dataframe.columns:
        result = adfuller(dataframe[column])
        p_value = result[1]
        p_values.append(p_value)

    results_df = pd.DataFrame({'Variable': dataframe.columns, 'p_value': p_values})

    return results_df


# In[54]:


adf_matrix=adf_test(df)
adf_matrix.head()


# In[63]:


(adf_matrix.p_value>0.05).sum()


# Todas las variables son estacionarias.
