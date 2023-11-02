#!/usr/bin/env python
# coding: utf-8

# # Codigo reciclado
# Tiene sentido revisar todo esto? Por lo menos en esta instancia me parece que NO, salvo que me lo pidan.

# ## Methodology:
# Quiero contestar:  
# * Hay autocorrelacion clusterizada de la varianza? Puedo verlo:
#     * Graficamente ACF y PACF
#     * Por tests ACF y PACF
# * Hay no normalidad? 
#     * como se comporta en las colas?
# 

# ## Startup

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

import os
import pickle


# In[2]:


np.random.seed(42)


# In[14]:


dataroute=os.path.join("..",  "data")
resultsroute=os.path.join("..",  "results", "descriptive")


# ## Data Retrieval

# In[15]:


start='2013-01-01'
end="2023-06-01"

name=f'processed_dataset_{start}_{end}.pickle'
filename=os.path.join(dataroute, name)

with open(filename, 'rb') as handle:
    data=pickle.load(handle)


# In[16]:


sb.set_style(style='darkgrid')
sb.set_palette(sb.color_palette(palette='deep'))


# In[17]:


def plot_vol(data, name):
    data[name]['gk_vol'].plot(ax=ax1)
    ax1.grid(True)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Variance')
    ax1.set_title(f'Garman-Klass intraday Volatility for {name}')
    fig.savefig(route_graphs + f'{name}_variance.png')


# In[ ]:





# In[18]:


import pandas as pd
import numpy as np
import pandas_datareader as web

import scipy.stats as scs
from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf

import datetime as dt
import seaborn as sb

return_lists = ['Returns', 'Log Returns', 'Abs Returns', 'Sqr Returns']


def autocorrelograms(df, names, lags):
    """
    :param df: diccionario con dataframes adentro
    :param list: lista con los #n nombres de las acciones que vamos a calcular
    :return:guarda #n graficos de autocorrelacion en la ruta seleccionada
    """
    for name in names:
        fig = plt.figure(figsize=[13.6, 10.2])
        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
        ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)

        plot_acf(df[name]['Log Returns'],
                 lags=lags,                 # Cuantos lags busco autocorrelacionar
                 zero=False,                # Si tomo el lag cero
                 alpha=0.05,                # El rango de certeza marcado en azul
                 use_vlines=False,          # Lineas verticales que conectan cada punto con el eje x
                 ax=ax1)                    # La posicion en la figura
        ax1.grid(True)
        ax1.set_xlabel('Lag')
        ax1.set_ylabel('Autocorrelation')
        ax1.set_title(f'Autocorrelation of Log returns for {name}')

        plot_acf(df[name]['Abs Returns'],
                 lags=lags,                 # Cuantos lags busco autocorrelacionar
                 zero=False,                # Si tomo el lag cero
                 alpha=0.05,                # El rango de certeza marcado en azul
                 use_vlines=False,          # Lineas verticales que conectan cada punto con el eje x
                 ax=ax2)                    # La posicion en la figura
        ax2.grid(True)
        ax2.set_xlabel('Lag')
        ax2.set_ylabel('Autocorrelation')
        ax2.set_title(f'Autocorrelation of Abs returns for {name}')

        plot_acf(df[name]['Sqr Returns'],
                 lags=lags,                 # Cuantos lags busco autocorrelacionar
                 zero=False,                # Si tomo el lag cero
                 alpha=0.05,                # El rango de certeza marcado en azul
                 use_vlines=False,          # Lineas verticales que conectan cada punto con el eje x
                 ax=ax3)                    # La posicion en la figura
        ax3.grid(True)
        ax3.set_xlabel('Lag')
        ax3.set_ylabel('Autocorrelation')
        ax3.set_title(f'Autocorrelation of Sqr returns for {name}')

        fig.savefig(route_graphs + f'{name}_autocorrs_returns.png')

# autocorrelograms(dfs, stocks, 252)

def ac_test(df, names, lags):
    "FINALMENTE NO USE ESTA FUNCION"
    ac_list = []
    ac_pvals = []
    pac_list = []

    for rets in return_lists:
        for name in names:
            ac, confint, qstat, pvals = acf(df[name][rets],nlags=lags, qstat=True, alpha=0.05)
            ac_list.append(np.round(ac[:5],3))
            ac_pvals.append(pvals[:5])

        ac_df = pd.DataFrame(ac_list)
        ac_df.index = names
        ac_df.to_csv(route_tables + f'{rets}_ac_table.csv')

        ac_p_df = pd.DataFrame(np.round(ac_pvals,3))
        ac_p_df.index = names
        ac_p_df.to_csv(route_tables + f'{rets}_ac_pval_table.csv')

        ac_list = []
        ac_pvals = []

    return ac_df, ac_p_df

# ac_test(dfs, stocks, 252)

def pac_test(df, names, lags):
    "FINALMENTE NO USE ESTA FUNCION"
    pac_list = []

    for rets in return_lists:
        for name in names:
            pac = pacf(df[name][rets], nlags=lags)
            pac_list.append(np.round(pac[:5], 3))

        pac_df = pd.DataFrame(pac_list)
        pac_df.index = names
        pac_df.to_csv(route_tables + f'{rets}_pac_table.csv')

        pac_list=[]

    return pac_df

# pac_test(dfs, stocks, 252)


def partial_autocorrelograms(df, names, lags):
    """

    :param df: diccionario con dataframes adentro
    :param list: lista con los #n nombres de las acciones que vamos a calcular
    :return:guarda #n graficos de autocorrelacion en la ruta seleccionada
    """
    for name in names:
        fig = plt.figure(figsize=[13.6, 10.2])
        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
        ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)

        plot_pacf(df[name]['Log Returns'],
                  method='ywunbiased',          # Metodo de Yule Walker con correccion de sesgo por autocovarianzas
                  lags=lags,                    # Cuantos lags busco autocorrelacionar
                  zero=False,                   # Si tomo el lag cero
                  alpha=0.05,                   # El rango de certeza marcado en azul
                  use_vlines=False,             # Lineas verticales que conectan cada punto con el eje x
                  ax=ax1)                       # La posicion en la figura
        ax1.grid(True)
        ax1.set_xlabel('Lag')
        ax1.set_ylabel('Partial Autocorrelation')
        ax1.set_title(f'Partial Autocorrelation of Log returns for {name}')

        plot_pacf(df[name]['Abs Returns'],
                  method='ywunbiased',          # Metodo de Yule Walker con correccion de sesgo por autocovarianzas
                  lags=lags,                    # Cuantos lags busco autocorrelacionar
                  zero=False,                   # Si tomo el lag cero
                  alpha=0.05,                   # El rango de certeza marcado en azul
                  use_vlines=False,             # Lineas verticales que conectan cada punto con el eje x
                  ax=ax2)                       # La posicion en la figura
        ax2.grid(True)
        ax2.set_xlabel('Lag')
        ax2.set_ylabel('Partial Autocorrelation')
        ax2.set_title(f'Partial Autocorrelation of Abs returns for {name}')
        
        plot_pacf(df[name]['Sqr Returns'],
                  method='ywunbiased',          # Metodo de Yule Walker con correccion de sesgo por autocovarianzas
                  lags=lags,                    # Cuantos lags busco autocorrelacionar
                  zero=False,                   # Si tomo el lag cero
                  alpha=0.05,                   # El rango de certeza marcado en azul
                  use_vlines=False,             # Lineas verticales que conectan cada punto con el eje x
                  ax=ax3)                       # La posicion en la figura
        ax3.grid(True)
        ax3.set_xlabel('Lag')
        ax3.set_ylabel('Partial Autocorrelation')
        ax3.set_title(f'Partial Autocorrelation of Sqr returns for {name}')
        
        fig.savefig(route_graphs + f'{name}_partial_autocorrs_returns.png')

# partial_autocorrelograms(dfs,stocks,252)

def histograms(df, names):
    for name in names:
        fig = plt.figure(figsize=[13.6, 10.2])
        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2)
        ax3 = fig.add_subplot(3, 1, 3)

        sb.distplot(df[name]['Log Returns'].fillna(0),
                    ax=ax1,
                    hist=True,
                    bins=int(np.ceil(np.log2(len(df[name])) + 15)),
                    label='Datos observados',
                    kde=True,
                    kde_kws={"color":"k", "lw":2, "label":"KDE"},
                    fit=norm,
                    fit_kws = {"color":"r", "lw":3, "label":"Normal Teorica"})
        # TODO: No me esta dando que las frecuencias relativas esten ni cerca de lo esperable
        plt.grid(True)
        plt.xlabel('Log Returns')
        plt.ylabel('Frequency')
        #plt.legend(True)
        plt.title(f'Histogram for Log returns frequency for {name}')

        sb.distplot(df[name]['Abs Returns'].fillna(0),
                    ax=ax2,
                    hist=True,
                    bins=int(np.ceil(np.log2(len(df[name])) + 15)),
                    label='Datos observados')
                    # SAQUE LO DE ABAJO PQ FLASHIE
                    # kde=True,
                    # kde_kws={"color":"k", "lw":2, "label":"KDE"},
                    # fit=halfnorm,
                    # fit_kws = {"color":"r", "lw":3, "label":"Media Normal Teorica"})

        plt.grid(True)
        plt.xlabel('Abs Returns')
        plt.ylabel('Frequency')
        #plt.legend(True)
        plt.title(f'Histogram for Abs Returns frequency for {name}')

        sb.distplot(df[name]['Sqr Returns'].fillna(0),
                    ax=ax3,
                    hist=True,
                    bins=int(np.ceil(np.log2(len(df[name])) + 15)),
                    label='Datos observados')
                    # kde=True,
                    # kde_kws={"color":"k", "lw":2, "label":"KDE"},
                    # fit=chi2,
                    # fit_kws = {"color":"r", "lw":3, "label":"Chi Cuadrada Teorica"})

        plt.grid(True)
        plt.xlabel('Sqr Returns')
        plt.ylabel('Frequency')
        #plt.legend(True)
        plt.title(f'Histogram for Sqr Returns frequency for {name}')

        fig.savefig(route_graphs + f'{name}_histogram_returns.png')

# histograms(dfs, stocks)
def histogram_normal(df, names):
    for name in names:
        fig = plt.figure(figsize=[13.6, 5.1])
        ax1 = fig.add_subplot(1, 1, 1)

        sb.distplot(df[name]['Returns'].fillna(0),
                    ax=ax1,
                    hist=True,
                    bins=int(np.ceil(np.log2(len(df[name])) + 15)),
                    label='Datos observados',
                    fit=norm,
                    fit_kws = {"color":"r", "lw":3, "label":"Normal Teorica"})

        plt.grid(True)
        plt.xlabel('Log Returns')
        plt.ylabel('Frequency')
        #plt.legend(True)
        plt.title(f'Histogram for simple return frequency for {name}')
        fig.savefig(route_graphs + f'{name}_normality_histogram_returns.png')

# histogram_normal(dfs, stocks)

def normality_test(arr):
    arr = arr.fillna(0)
    print('Skewness coefficient: ' + str(np.round(scs.skew(arr), 2)))
    print('Skewness test p-value: ' + str(1 - np.round(scs.skewtest(arr)[1], 2)))
    print('Kurtosis coefficient: ' + str(np.round(scs.kurtosis(arr), 2)))
    print('Kurtosis test p-value: ' + str(1 - np.round(scs.kurtosistest(arr)[1], 2)))
    print('Normality test p-value: ' + str(1 - np.round(scs.normaltest(arr)[1], 2)))

def normality_table(df, names, values):
    """
    :param df: 
    :param names: 
    :return: 
    """
    skew = []
    skew_pval = []
    kurt = []
    kurt_pval = []
    norm_pval = []

    for name in names:
        skew.append(np.round(scs.skew(df[name][values]), 3))
        skew_pval.append(np.round(scs.skewtest(df[name][values])[1], 3))
        kurt.append(np.round(scs.kurtosis(df[name][values]), 3))
        kurt_pval.append(np.round(scs.kurtosistest(df[name][values])[1], 3))
        norm_pval.append(np.round(scs.normaltest(df[name][values])[1], 3))

    dictionary = {'Skewness':skew, 'Skew p-value':skew_pval, 'Kurtosis':kurt, 'Kurtosis p-value':kurt_pval,
                  'Normality test p-value':norm_pval}

    table = pd.DataFrame(dictionary)
    table.index = names
    table.to_csv(route_tables+f'norm_table_{values}.csv')

    return table

# for rets in return_lists:
#     normality_table(dfs, stocks, rets)


def describe_and_test_norm(df, names, values):
    "FINALMENTE NO USÈ ESTA FUNICON"
    for name in names:
        print(values + ' tests for ' + name)
        print(df[name][values].describe())
        print('')
        print(values+' tests for '+name)
        normality_test(df[name][values])
        print('')
        print('-' * 10)

    table = normality_table(df, names, values)

    return table
            

# describe_and_test_norm(dfs, stocks, 'Returns')


""

## para tests de residuales
def residuals_test_plot(residual, lags, alpha_graph=0.05,**kwargs):
    """
    TODO: FUNCION SIN USO AUN.
    :param residual: The model Residual we want to test
    :param lags: the amount of lags we want to run the ACF/PACF on and the lags on the Ljung-Box test
    :param alpha_graph: The significance of the graph region
    :param name: Name for the graph
    :return: the p-value tuple and a saved ACF/PACF graph
    """
    fig = plt.Figure(figsize=[10.2, 13.6])
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    plot_acf(residual, lags=lags, zero=False, alpha=alpha_graph, ax=ax1)
    plot_pacf(residual, lags=lags, zero=False, alpha=alpha_graph, ax=ax2)
    plt.show()

    name = kwargs.get('name', 'model')
    fig.savefig(f'{name}_ACF_PACF')

    test = acorr_ljungbox(residual, lags=lags)

    print('The p-values for the residuals are:')
    print(test[1])


# In[ ]:




