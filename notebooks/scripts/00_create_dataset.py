#!/usr/bin/env python
# coding: utf-8

# ## Startup

# In[1]:


import numpy as np
import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt

import os
import pickle


# In[2]:


import yfinance as yf
yf.pdr_override()


# In[3]:


np.random.seed(42)


# In[4]:


dataroute=os.path.join("..",  "data")
dumproute=os.path.join("..",  "dump")
resultsroute=os.path.join("..",  "results")


# ## Data Retrieval

# In[5]:


tickerlist=["^MERV", 
            "GGAL", "GGAL.BA", 
            "YPF", "YPFD.BA",
            "EDN", "EDN.BA",
            "BMA", "BMA.BA"] 
# sumar tamb BBAR/BBAR? TEO/TECO2?


# In[6]:


with open(os.path.join(dumproute, "tickerlist.pickle"), 'wb') as f:
          pickle.dump(tickerlist, f, protocol=pickle.HIGHEST_PROTOCOL)
          # es esta la lista que realmente necesito? luego aparecen USD y ^MERV_USD


# In[7]:


factordict={"GGAL": 10, "YPF":1, "EDN":20, "BMA":10, "BBAR":3, "TEO":5}


# In[8]:


stocks=tickerlist.copy()
stocks.remove("^MERV")
stocklist=[]

for i in range(0, len(stocks), 2):
    stocklist.append((stocks[i], stocks[i+1]))
del stocks
stocklist


# In[9]:


ohlclist=["Open", "High", "Low", "Close"]


# In[10]:


objectlist=[]

for ticker in tickerlist:
    objectlist.append(yf.Ticker(ticker))    


# In[11]:


# get historical market data
data={}
start='2013-01-01'
end="2023-06-01"


# In[12]:


name=f'dataset_{start}_{end}.pickle'
filename=os.path.join(dataroute, name)


# In[13]:


if not os.path.exists(filename):
    for ticker in objectlist:
        # descargo data en un diccionario[ticker]
        data[ticker.ticker] = ticker.history(start=start, end=end)
        # guardo en un pickle
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
else:
    with open(filename, 'rb') as handle:
        data=pickle.load(handle)


# ## Data quality deletion

# In[14]:


data_quality_dates=["2022-07-14"]


# In[15]:


for ticker in tickerlist:
    data[ticker]=data[ticker].loc[~data[ticker].index.isin(pd.to_datetime(data_quality_dates))]


# ## Implicit USD calculation

# In[16]:


def _reindex_refill_dfs(df1, df2):
    """
    The function returns two dataframes with an index as the union of the two.
    The dataframes are then forward filled.
    """
    index3=df1.index.union(df2.index)
    # reindex both con index3
    df3=df1.reindex(index3)
    df4=df2.reindex(index3)
    # fillna con previous value
    df3.fillna(method="ffill")
    df4.fillna(method="ffill")
    return df3, df4


# In[17]:


def calculate_usd(usd_df, ars_df, conversion_factor):
    """
    The function returns a dataframe with an index the size of the union between the two.
    Missing values in dates (stemming from, for example, holidays in one country) are
    forward filled to create the last  
    """
    usd_df_r, ars_df_r = _reindex_refill_dfs(usd_df, ars_df)
    implicit_usd = ars_df_r.divide(usd_df_r)*conversion_factor
    return implicit_usd


# In[18]:


usdlist=[]
for stocktuplo in stocklist:
    us, ba = stocktuplo
    usdlist.append(f"USD_{us}")
    data[f"USD_{us}"]=calculate_usd(data[us][ohlclist], data[ba][ohlclist], factordict[us])
    data[f"USD_{us}"]["Average"]=data[f"USD_{us}"].mean(axis=1)


# In[19]:


data["USD"]=pd.DataFrame(columns=ohlclist)

for i in ohlclist:
    df=pd.concat([data[col][i] for col in usdlist], axis=1)
    data["USD"][i]=df.mean(axis=1)
    
data["USD"]["Average"]=data["USD"].mean(axis=1)


# In[20]:


for key in data.keys():
    data[key].fillna(method="ffill", inplace=True)
    # revisar esto


# In[21]:


data["USD"][[*ohlclist, "Average"]].plot(figsize=(10,10), logy=True, grid=True)


# ## USD Denominated Index

# In[22]:


data["USD_^MERV"]=pd.DataFrame(columns=ohlclist)

for col in ohlclist:
    data["USD_^MERV"][col] = data["^MERV"][col]/data["USD"]["Average"]


# In[23]:


data["USD_^MERV"].fillna(method="ffill", inplace=True)


# ## Intraday Volatility

# Vamos a usar para medir intraday volatility el estimador de Garman and Klass (1980):
# 
# $$V_{ohlc}=0.5*[log(H)-log(L)]^2+(2*log(2)-1)*[log(C)-log(O)]^2$$ 
# Donde H es el precio mas alto del día, L el bajo, C el cierre y O su apertura
# 
# Garman, M. B. and M. J. Klass (1980). On the estimation of security price volatilities from historical data. Journal of Business 53, 67–78.

# In[24]:


def gk_vol(o, h, l, c):
    "Returns Garman Klass (1980) intraday volatility estimator"
    return 0.5*(np.log(h)-np.log(l))**2+(2*np.log(2)-1)*(np.log(c)-np.log(o))**2


# ## Returns Calculation

# In[25]:


for ticker in data.keys():
    view=data[ticker]
    view["rets"] = view["Close"]/view["Close"].shift()-1
    view["log_rets"] = np.log(view["Close"]/view["Close"].shift())
    view["norm_range"] = (view["High"]-view["Low"])/view["Open"]
    # chequear si esto tiene asidero
    # alternativa (view["High"]-view["Low"])/view["Close"]
    view["gk_vol"] = gk_vol(o=view["Open"], h=view["High"], l=view["Low"], c=view["Close"])
    # delete first observation to eliminate nans
    data[ticker]=data[ticker][1:].copy()


# ## Process into single dataframe, matching dates and forward filling
# Véase https://github.com/alfsn/regime-switching-hmm/issues/9

# In[26]:


df=pd.DataFrame()

for key, value in data.items():
    for column in ["rets", "log_rets", "gk_vol"]:
        df[key+"_"+column]=value[column]


# In[27]:


df.loc[df.isna().any(axis=1), df.isna().any(axis=0)]


# In[28]:


df.fillna(0, inplace=True)


# ## Excluimos los dólares implícitos

# In[29]:


usdlist=[]
for key in data.keys():
    if key.startswith("USD"):
        usdlist.append(key)
usdlist.remove("USD")
usdlist.remove("USD_^MERV")        

print(usdlist)

for col in usdlist:
    del data[col]


# ## Save dataset

# In[30]:


processedname="processed_"+name
with open(os.path.join(dataroute, processedname), 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[31]:


particular_USDs=[column for column in df.columns if ((column.startswith("USD")) and ("^MERV" not in column))]
particular_USDs.remove("USD_rets") 
particular_USDs.remove("USD_log_rets")
particular_USDs.remove("USD_gk_vol")
particular_USDs


# In[32]:


df_clean= df.drop(columns=particular_USDs)
df_clean


# In[33]:


assert not (df_clean.isna()).any().any(), "Existen n/a"


# In[34]:


finaldfname="finaldf_"+name
with open(os.path.join(dataroute, finaldfname), 'wb') as handle:
    pickle.dump(df_clean, handle, protocol=pickle.HIGHEST_PROTOCOL)

