#!/usr/bin/env python
# coding: utf-8

# ## Startup

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import arch

import os
import pickle
import warnings


# In[2]:


np.random.seed(42)


# In[3]:


from scripts.params import get_params

params = get_params()


# In[4]:


dataroute = os.path.join("..", "data")
processedroute = os.path.join("...", "processed")
resultsroute = os.path.join("..", "results")


# ## Data Retrieval

# In[5]:


name = f'finaldf_train_{params["tablename"]}.pickle'
filename = os.path.join(dataroute, name)
with open(filename, "rb") as handle:
    df = pickle.load(handle)


# In[6]:


df.head()


# ## GARCH Training
# Warning: this section only uses log_rets as y variables. See:
# https://github.com/alfsn/regime-switching-hmm/issues/35

# In[7]:


# Define the range of p and q values
alpha_values = [1, 2, 3, 4]  # estos son los valores de los lags in mean del AR.
p_values = [1, 2, 3]  # Example: p values
q_values = [0, 1, 2, 3]  # Example: q values
# all models with q=0 are exclusively ARCH (non-GARCH)


# In[8]:


models = {}
predict = {}


# In[9]:


best_aic = {}
best_bic = {}


# In[10]:


def check_best_aic(key, model, previous_best: float, p: int, q: int, dist: str):
    """
    AIC is better when lower.
    """
    if model == None:
        pass
    else:
        if model.aic < previous_best:
            best_aic[key] = {
                "model": model,
                "aic": model.aic,
                "p": p,
                "q": q,
                "dist": dist,
            }


# In[11]:


def check_best_bic(key, model, previous_best: float, p: int, q: int, dist: str):
    """
    BIC is better when lower.
    """
    if model == None:
        pass
    else:
        if model.aic < previous_best:
            best_bic[key] = {
                "model": model,
                "bic": model.bic,
                "p": p,
                "q": q,
                "dist": dist,
            }


# In[12]:


# Estimate ARMA-ARCH and ARMA-GARCH models for different p and q values
nonconverged_models = 0
ok_models = 0

for key in params["tickerlist"]:
    returns = df[f"{key}_log_rets"]

    models[key] = {}
    predict[key] = {}

    best_aic[key] = {"aic": np.inf}
    best_bic[key] = {"bic": np.inf}

    for p in p_values:
        for q in q_values:
            for dist in ["Normal", "StudentsT"]:
                model = arch.arch_model(
                    returns,
                    mean="AR",
                    lags=1,
                    vol="Garch",
                    p=p,
                    q=q,
                    dist=dist,
                    rescale=False,
                )
                results = model.fit(
                    options={"maxiter": 2000}, disp="off", show_warning=False
                )

                if results.convergence_flag != 0:
                    # 0 is converged successfully
                    # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_slsqp.html
                    results = None
                    nonconverged_models += 1
                else:
                    ok_models += 1

                check_best_aic(
                    key=key,
                    model=results,
                    previous_best=best_aic[key]["aic"],
                    p=p,
                    q=q,
                    dist=dist,
                )
                check_best_bic(
                    key=key,
                    model=results,
                    previous_best=best_bic[key]["bic"],
                    p=p,
                    q=q,
                    dist=dist,
                )

                models[key][(p, q, dist)] = results

print()
print(f"ok: {ok_models}")
print(f"nonconverged: {nonconverged_models}")


# # Residuals

# In[13]:


name = f'finaldf_test_{params["tablename"]}.pickle'
filename = os.path.join(dataroute, name)
with open(filename, "rb") as handle:
    df_test = pickle.load(handle)


# In[14]:


def generate_GARCH_samples_residuals(
    model_dict: dict, insample_data: pd.DataFrame, oos_data: pd.DataFrame
):
    """
    Esta función come archmodelresults (que vienen del diccionario best_aic y best_bic),
    y hace pronósticos rolling (con ventana de 1 año (252 días habiles)),
    lo que devuelve samples y residuos.
    El método de pronóstico es de simulación

    Args:
        model_dict (_type_): _description_
        pd (_type_): _description_

    Returns:
        _type_: _description_
    """
    split_date = oos_data.index[0]
    dates_to_forecast = len(oos_data.index)

    oos_data = pd.concat([insample_data, oos_data])
    del insample_data

    # vamos a implementar recursive window forecasting
    # https://arch.readthedocs.io/en/latest/univariate/forecasting.html
    # https://arch.readthedocs.io/en/latest/univariate/univariate_volatility_forecasting.html#Recursive-Forecast-Generation

    index = oos_data.index
    end_loc = np.where(index >= split_date)[0].min()
    # esto es un int del iloc
    # preciso usar ints de iloc porque el timedelta se me va a romper con el fin de semana
    rolling_window = 252

    forecasts = {}

    model = arch.arch_model(
        y=oos_data,
        mean="AR",
        lags=1,
        vol="Garch",
        p=model_dict["p"],
        q=model_dict["q"],
        dist=model_dict["dist"],
        rescale=False,
    )

    for i in range(1, dates_to_forecast):
        date_of_first_forecast = oos_data.index[end_loc + i]

        res = model.fit(
            first_obs=end_loc - rolling_window + i, last_obs=end_loc + i, disp="off"
        )

        forecast = res.forecast(
            horizon=1, start=date_of_first_forecast, method="simulation"
        ).mean.iloc[0]

        forecasts[forecast.name] = forecast

    forecasts = pd.DataFrame(forecasts).T
    forecasts.columns = oos_data.columns

    pct_nan = forecasts.iloc[:, 0].isna().sum() / len(forecasts.index) * 100

    if pct_nan > 5:
        warnings.warn(f"{oos_data.columns[0]} % na: {pct_nan}")

    forecasts.fillna(method="ffill", inplace=True)

    residuals = oos_data - forecasts

    return forecasts, residuals


# In[15]:


def save_as_pickle(data, criterion: str, type_save: str):    
    with open(
        os.path.join(
            resultsroute,
            f"""GARCH_{params["tablename"]}_{criterion}_best_{type_save}.pickle""",
        ),
        "wb",
    ) as output_file:
        pickle.dump(data, output_file)


# In[16]:


forecasts_dict={"aic":{}, "bic":{}}
residuals_dict={"aic":{}, "bic":{}}

for criterion, dictionary in zip(["aic", "bic"], [best_aic, best_bic]):
    for stock in dictionary.keys():
        forecasts, residuals = generate_GARCH_samples_residuals(
            dictionary[stock],
            pd.DataFrame(df[f"{stock}_log_rets"]),
            pd.DataFrame(df_test[f"{stock}_log_rets"])
            )

        forecasts_dict[criterion][stock]=forecasts
        residuals_dict[criterion][stock]=residuals     


# In[17]:


for criterion, bestmodels in zip(["aic", "bic"],[best_aic, best_bic]):
    save_as_pickle(forecasts_dict[criterion], criterion=criterion, type_save="forecasts")
    save_as_pickle(residuals_dict[criterion], criterion=criterion, type_save="residuals")
    save_as_pickle(bestmodels, criterion=criterion, type_save="models")


# # Plotting
# ## TODO: Esto aun está feo: tengo que armar que esto devuelva el plotteo de returns y los predicts uno encima del otro

# In[18]:


def plot_close_rets(data, model, key, name):
    fig = plt.figure(figsize=(20, 20))
    plt.tight_layout()
    plt.title(f"{key} Log returns")

    plt.subplot(1, 1, 1)

    x = data[key]["log_rets"]
    y = data[key].index

    plt.plot(x, y, ".", c="red")
    # plt.plot(x, model.predict(x), '.', c="blue")

    plt.grid(True)
    plt.xlabel("datetime", fontsize=16)
    plt.ylabel("log rets", fontsize=16)

    plt.savefig(
        os.path.join(resultsroute, "graphs", f"GARCH", f"{key}_model_{name}.png")
    )


# In[19]:


# for key in data.keys():
#    print(key)
#    plot_close_rets(data, key)
# plt.show()

