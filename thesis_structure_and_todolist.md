# Thesis structure and todo list
Abrimos este archivo con "control+k" y luego "v" en VSC

## Research Question

"To what extent can Hidden Markov Models (HMMs) outperform traditional time series models like VAR, ARMA-ARCH, and related models in forecasting one-day returns and volatility in financial markets?"

## Thesis Structure

### Chapter 1: Introduction

Background and motivation
Research question and objectives
Significance and contribution of the study
Outline of the thesis

### Chapter 2: Literature Review

Overview of financial time series forecasting
Introduction to VAR, ARMA, ARCH, and related models
Introduction to Hidden Markov Models (HMMs)
Previous studies on financial market forecasting
Gap identification

### Chapter 3: Data Collection and Preprocessing

Description of data sources
Data collection methods
Data preprocessing techniques

> Tiene sentido trabajar con otras medidas de vol?
Abs return
Square returns
Newfile: Descriptive stats

Stationarity tests
Volatility estimation

### Chapter 4: Methodology

Detailed explanation of VAR, ARMA, ARCH, and related models
Detailed explanation of Hidden Markov Models (HMMs)
Model parameter estimation
> ARMA-ARCH estimation
Revisar que pasa con p=0
Revisar rescaling: sirve log_rets*100? Luego me deja un modelo malentrenado?

> Multivariate HMM
> Change "model" dict from single modelname to model[key][comp] nested dict
> Create prediction dict to be pickled
> Create a log for each model, DO NOT PRINT
> Decide what to do with models that do not converge


Model selection criteria

> TODO:
Newfile: Model comparer
Think about how to port best HMM data and best ARMA-ARCH into a common comparison.

>Posiblemente requiera armar una funcion de chequeo que use model.monitor==True para chequear convergencia del modelo.  
Recien ahi rankear por AIC&BIC
https://hmmlearn.readthedocs.io/en/latest/api.html#hmmlearn.base.ConvergenceMonitor

Hybrid models (if applicable)

### Chapter 5: Empirical Analysis

Model implementation and selection
In-sample and out-of-sample testing
Evaluation metrics (e.g., Mean Absolute Error, Root Mean Squared Error)
Backtesting strategies
Interpretation of results

### Chapter 6: Comparative Analysis

Comparison of forecasting accuracy between VAR, ARMA-ARCH, and HMMs
Statistical tests (e.g., paired t-tests)
Analysis of model performance in different market conditions
Sensitivity analysis

### Chapter 7: Robustness and Sensitivity Analysis

Robustness tests (e.g., different data periods, alternative model specifications)
Sensitivity to parameter choices
Monte Carlo simulations (if relevant)

### Chapter 8: Discussion

Interpretation of results and implications
Limitations and challenges
Practical implications for financial markets
Future research directions

### Chapter 9: Conclusion

Summary of findings
Contribution to the field of financial time series forecasting
Policy and investment implications
Final remarks

### Chapter 10: References

Comprehensive list of all references used in the thesis
Appendices (if necessary)

Supplementary materials such as code, data descriptions, and additional results