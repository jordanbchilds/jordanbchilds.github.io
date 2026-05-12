---
title: "A Comparison of Statistical Models of Energy Spot Price in the UK"
date: 2026-05-12

mathjax: true
tags: ["Statistics", "Python"]
---

# Introduction

Unlike other commodities, such as oil, storing energy for later use is difficult and expensive. Therefore, it's supply and demand must be monitored real-time and alterations must be made to maintain an equilibrium. One of the ways in which this equilibrium is maintained is that energy production and consumption are predicted one day ahead, and an hourly energy spot-price required to close the deficit is calculated. The hourly spot-price is the price paid for energy to close the deficit between predicted and real-time us. The price is determined the day prior and so the today's spot-prices were set yesterday.

Being able to model the energy spot-price is, therefore, of keen interest to energy producers [AND OTHERS?] who would be able to profit by increasing or decreasing their supply if the price is high enough. In this blog post we investigate a range or time-series modelling techniques to be able to predict the energy spot-price and compare them based on their predictive performance.

# The Data

## Data source

The data is sourced from Transmission System Operators and power exchanges, and is available through [kaggle](https://www.kaggle.com/datasets/eugeniyosetrov/load-wind-and-solar-prices-in-hourly-resolution), a detailed description of collection methods and data structure is given on a relating [GitHub page](https://github.com/Open-Power-System-Data/time_series/blob/master/main.ipynb). The dataset is large and contains spot prices and explanatory variables for many regions within the Europe. The modelling here is limited to data within Great Britain, and the explanatory variables are reduced to those believed to be most important; solar generation, wind generation, and forecasted energy use. 

## Features of the data

Upon inspection of the data there appears to be little overall trend in energy spot price over the five year period, with relatively flat long-term mean, see Figure 1. However, there are significant price spike increasing or decreasing the price outside of its usual range. Most notably in late 2017, when the price spiked by many magnitudes. We also see a strong seasonal pattern in the data between days, with a high correlation for energy prices collected at the same time of day, see Figure 1. Prices on the weekend appear to differ from their weekday counter-parts but maintain a similar overall trend.

## Missing values

The dataset contains some missing values, however the degree of missingness is relatively low, with between 41 and 121 missing values per variable in a dataset of over 50,000 observations. The missing values are therefore not of concern in regards to data quality but should be filled to make model fitting easier and more consistent across modelling techniques and python packages. To estimate the missing values for each variable a Kalman filter is used to extract the smoothed estimates. The filter is fitted independently for each variable and the missing values replaced by their estimates. The `filled` data is used for all model fitting throughout this investigation, but the estimated values are omitted when calculating model comparison statistics, such as mean squared error.

<img src="/static/energy_spot_price_statistical_model_comparison/spot_price_data.png" alt="Raw data" width="1770" height="1466">

<img src="/static/energy_spot_price_statistical_model_comparison/energy_price_by_day.png" alt="By day price" width="2068" height="1166">

# Methods

Three model categories are compared, each one is considered with and without the use of explanatory (exogenous) variables, and their predictive performance is compared using a sliding window prediction. The energy spot-price is set in 24 hour chucks, that is, the hourly spot prices for tomorrow, for example, are set today. Making predictions 24 hours ahead replicates the real-world need for predicting hourly spot-prices the day before. The dataset is split into two subsets; training and validation, with the final month's worth of data being used to validate and compare model predictions. A sliding window is used to make predictions, reflecting the necessity to forecast the next days energy price, For each model the window used spans a two year period (assumed to be 730 days). Although not shown here, fitting the model on the log-scale provides a better fit to the data and is consistent with other works modelling energy prices and so all models considered are here are on the log-scale. 

For the models described in the this blog, the response variable, $Y_t$, denotes the log spot-price at time $t$. The set of observed exogenous variables observed at time $t$ are denoted $\boldsymbol{x}_{t}$. This set contains the wind and solar generation of the previous day, the forecasted energy load, and two dummy variables indicating a whether the observation is made on a Saturday or Sunday. The previous days wind and solar generation must be used as generation at the time of spot-price would not be known. 

In the remainder of this section, the different models are described and their implementation is briefly described. A detailed description of each model and its mathematical and statistical underpinnings are not provided here, but a short description is provided. 

## ARIMA

Auto-Regression Integrated Moving Average (ARIMA) models are ubiquitous within time-series modelling and combines two modelling techniques into a single model. An autoregression (AR) models the current state of a time-series as a linear combination of the previous $p$ states, while a moving average (MA) model models the current state as a linear combination of the previous $q$ model errors. Combing these, an ARMA$(p,q)$ models the current state as a linear combination of the previous $p$ states and $q$ model errors. The integrated component of the model arises from the ARMA assumption of stationarity i.e. there is no trend within the data. If a time series is not stationary, the response variable can be differenced $d$ times to achieve a stationary series. As well as stationarity, ARMA models assume constant model variance (volatility). The model is defined for a orders $(p,d,q)$ and can be written as 
<div class="math">
$$
\begin{aligned}
    &\;\phi(L)(1-L)^d Y_t = \theta(L)\epsilon_t, \\
    \phi(L) &= 1 - \phi_1 L - \Phi_2 L^{s} - \dots - \Phi_p L^{p}, \\
    \theta(L) &= 1 + \theta_1 L + \theta_2 L^{2} + \dots + \theta_q L^{q},\\
\end{aligned}
$$
</div>

where the operator $L$ is known as the lag operator and is defined such that $L^n Y_t = Y_{t-n}$. Although appearing complex, the model can be broken down into more easily digestible parts. Firstly, the differencing is achieved by the $(1-L)^d$ term, which acts directly on the observed data $Y_t$. The AR model component is achieved through $\left(1-\phi(L)\right)$, which acts on the stationary series produced through differencing. Expanding the brackets, it can be seen that this produces a linear combination of the previous $p$ differenced values. Lastly the MA component is achieved by the $\left(1+\theta(L)\right)$, which acts on the model errors, expanding the brackets we find this to be a linear combination of the previous $q$ model errors. 

The model can be extended to include seasonal effects and exogenous variables. Seasonal effects often have a significant impact on a time-series. For example, consider a energy generated via solar panels, which is highly correlated to the time of year. Including this seasonal relationship within a model can drastically improve model fit and performance. A seasonal component to an ARIMA model is introduced at a specified seasonal lag, $s$, and seasonal components are added multiplicatively such that interactions between seasonal and non-seasonal components arise within the model. Exogenous variables, which are believed to be correlated to the response variable, are included additively akin to a linear regression. For a time-series $Y_1, Y_2, \dots$ and set of $m$ exogenous variables observed at time $t$, $\boldsymbol{x}_{t}$, the general seasonal ARIMAX model is
<div class="math">
$$
\begin{aligned}
    \Phi(L^s)\phi(L)(1-&L)^d(1-L^s)^D Y_t = \Theta(L^s)\theta(L)\epsilon_t + \boldsymbol{b}^T\boldsymbol{x}_t,\\
    \Phi(L^s) &= 1 - \Phi_1 L^s - \Phi_2 L^{2s} - \dots - \Phi_P L^{Ps}, \\
    \phi(L) &= 1 - \phi_1 L - \Phi_2 L^{s} - \dots - \Phi_p L^{p}, \\
    \Theta(L^s) &= 1 + \Theta_1 L^s + \Theta_2 L^{2s} + \dots + \Theta_Q L^{Qs}, \\
    \theta(L) &= 1 + \theta_1 L + \theta_2 L^{2} + \dots + \theta_q L^{q}.
\end{aligned}
$$
</div>

Where $\boldsymbol{b}$ is $m$-dimensional vector of exogenous coefficients. Similarly to the ARIMA model, the may seem complex but each component of the model directly corresponds to a function and set of brackets, and expanding other brackets will yield a linear combination of the expected elements

### Model fitting

Here, we examine three versions of the ARIMA model; seasonal ARIMA, seasonal ARIMAX, and an hour-specific seasonal ARIMAX. For each model the fitting process follows the same steps. Firstly, the model orders are determined. This is a computationally intensive task and requires models with differing orders to be compared via their model fit. Due to computational constraints a subset of the training data was used, the final 100 days of the training set. Once the model orders are determined, by comparison of the Akaike information criteria (AIC), model predictions for the validation dataset were made via a sliding window using two years of data to predict the next 24 observations (one days worth of data). 

The seasonal order of the models is not inferred but fixed. During initial inspection of the data, a strong correlation was seen in the hour-specific energy price. As such, a daily seasonality is imposed for the seasonal ARIMA and ARIMAX models. The hour-specific ARIMAX model will include daily effects within it's non-seasonal component and so a weekly seasonal effect is imposed for this model. Models parameters are inferred via maximum likelihood estimation using the `auto_arima` and `SARIMAX` functions in python. The `SARIMAX` function is used to make out-of-sample predictions (forecasts). 

#### Seasonal ARIMA and Seasonal ARIMAX

The general form of the seasonal ARIMAX model if given above, and so formal definition is not necessary here. The mathematical definition of the seasonal ARIMA is found by removing the exogenous variables from the ARIMAX definition, and, again, a formal definite is omitted.

<img src="/static/energy_spot_price_statistical_model_comparison/arima_predictions.png" alt="ARIMA predictions" width="1764" height="1466">


#### Hourly independence seasonal ARIMA

The hourly-independent model, ARIMAt, differs slightly in its form, where observations made on different hours are assumed to be from an hour-specific seasonal ARIMA model. Let $Y_{t,h}$ be the logged energy spot-price on the $t$-th day and $h$-th hour. It is assumed that each model has the same seasonal effect, however the orders; $p,q,P,Q$, are allowed to vary. The model is defined as 
<div class="math">
$$
\begin{aligned}
    \Phi^h(L^s)&\phi^h(L)(1-L)^{d_h}(1-L^s)^{D_h} Y_{t,h}= \Theta^h(L^s)\theta^h(L)\epsilon_{t,h}\\
    \Phi^h(L^s) &= 1 - \Phi_1^h L^s - \Phi_2^h L^{2s} - \dots - \Phi_{P_h}^h L^{P_hs}\\
    \phi^h(L) &= 1 - \phi_1^h L - \Phi_2^h L^{s} - \dots - \Phi_{p_h}^h L^{p_h}\\
    \Theta^h(L^s) &= 1+  \Theta_1^h L^s + \Theta_2^h L^{2s} + \dots + \Theta_{Q_h}^h L^{Q_hs}\\
    \theta^h(L) &= 1 + \theta_1^h L + \theta_2^h L^{2} + \dots + \theta_{q_h}^h L^{q_h}\\
\end{aligned}
$$
</div>

<img src="/static/energy_spot_price_statistical_model_comparison/arimat_predictions.png" alt="sARIMAt predictions" width="1764" height="1466">

## VAR

The seasonal ARIMA presented consider energy prices as a single univariate time-series. However, the process could be viewed in a multivariate context where energy prices over a single day are treated as a random vector. This also reflects the modelling situation of day-ahead energy spot-prices, where a single 24-hour period is forecasted. Multivariate extensions of the ARIMA model exist, however, these are notoriously difficult to fit due, in part, to the increased number of model parameters, which scales quadratically with the number of lags, compared to linearly in the univariate setting. Much like the univariate case, the vector autoregressive (VAR) model assumes the current state depends on a linear combination of past states up to a defined lag, $p$. Let $\boldsymbol{Y}_{t}$ be a $k$-dimensional vector of log spot-prices for the $t$-th day i.e. $k=24$, the model is defined as
<div class="math">
$$
\begin{aligned}
    \boldsymbol{Y}_t &= \sum_{i=1}^p A_i \boldsymbol{Y}_{t-i} + B^T X_t + \boldsymbol{\epsilon}_t, \\
	\boldsymbol{\epsilon}_t &\sim \mathcal{N}\left(\boldsymbol{0}, \Sigma\right).
\end{aligned}
$$
</div>

Where $\boldsymbol{\epsilon}_{t}$ are independent and identically distributed multivariate normal random variables with zero-vector mean and covariance matrix $\Sigma$ and $A_i$ are $k\times k$ coefficient matrices. Exogenous variables are also introduced additively, for a $k\times m$ observed data matrix, the $k\times m$ coefficient matrix is denoted $B$. Similarly to ARIMA models, the VAR models assumes that volatility is constant. The VAR model is fitted to the logged spot-price data both with and without the inclusion of the exogenous variables.

### Model Fitting

[FUNCTIONS AND PREDICTION]

<img src="/static/energy_spot_price_statistical_model_comparison/var_predictions.png" alt="var predictions" width="1727" height="1466">


## ARCH

The ARIMA model (and extensions of) assume that the volatility is constant, this is likely not the case. Spikes in energy prices are known to occur due to a variety of political or practical factors, by their definition the spikes are outside of what would be expected from under normal circumstances and violate the constant volatility assumption. Autoregressive conditional heteroskedasticity (ARCH) models assume that the volatility is stochastic and dependent on a linear combination of the previous model errors upto a specified lag, $p$. The generalised ARCH (GARCH) models extends this to include a dependents on a linear combination of previous model variances, upto a specific lag, $q$. For a response variable $Y_t$, observed at time $t$, time-dependent model volatility is denoted $\sigma_t^2$. The model is defined as follows
<div class="math">
$$
\begin{aligned}
    Y_t &= \mu + \epsilon_t, \\
    \epsilon_t &\sim \text{Normal}\big( 0, \sigma_t^2 \big), \\
    \sigma_t^2 &= \alpha_0 + \sum_{i=1}^q \alpha_i\epsilon_{t-i}^2 + \sum_{i=1}^p\beta_i\sigma_{t-1}^2.
\end{aligned}
$$
</div>

The parameter $\mu$ indicates a consent overall mean from which random deviations are made. Although a useful, the model's assumption of a constant mean may be too strong and we wish to incorporate way to allow this vary in some sensible way.

### GARCH-ARX

The GARCH-ARX combines the GARCH model with an autoregression component to model the expected value of $Y_t$, rather than replying on a constant mean parameter. The mean, is therefore, modelled as linear combination of previous observations up to specified lag, $p^\prime$. Exogenous can also be introduced in a similar way manner to the ARIMA models. Continuing the notation from the discussion on ARIMA models, the GARCH-AR model is

<div class="math">
$$
\begin{aligned}
    &\phi(L)Y_t =  \boldsymbol{b}^T\boldsymbol{x}_t + \epsilon_t, \\
    \epsilon_t &\sim \text{Normal}\big( 0, \sigma_t^2 \big), \\
    \sigma_t &= \alpha_0 + \sum_{i=1}^q \alpha_i\epsilon_{t-i} + \sum_{i=1}^p\beta_i\sigma_{t-i}, \\
    \phi(L) &= 1 - \phi_1L^{1} - \phi_2L^{2} - \dots - \phi_{p^\prime}L^{p^\prime}. 
\end{aligned}
$$
</div>

Here two GARCH models are considered, both with AR mean models and one which includes exogenous variables.

### Model Fitting

The orders of the GARCH-AR models are chosen by fitting a range of models to the training dataset and choosing the parameter set which minimises the AIC. For the AR component a range of orders are tested which combine standard and seasonal lags, a seasonal lag of 24 is assumed.

[FUNCTIONS AND PREDICTIONS]

<img src="/static/energy_spot_price_statistical_model_comparison/garch_predictions.png" alt="GARCH predictions" width="1764" height="1466">

# Results

## Model fit and predictive performance

To choose the most appropriate model for energy spot-prices two aspects are considered; within-sample model fit and predictive power. To compare within-sample model fit we look at the AIC, which combines the likelihood penalised by the number of model parameters, favouring simpler models with fewer parameters. As well as, mean absolute error (MAE) between the fitted values (expected values) and observations as well as the root mean squared error (RMSE) which again compares the expected values to the observed data but penalises large errors more heavily. The resulting comparison statistics for all models are seen in Table 1. 

It is clear from the AIC statistics that the sARIMA- and GARCH-type models are preferred over the sARIMAt and VAR models, showing drastically lower AICs. The difference in their values is likely due to the increase in the number of model parameters, as the MAE and RSME for models are fairly similarly, suggesting a comparable fits to the data. Recall that the VAR models have large number of parameters, requiring a $24\times 24$ coefficient matrix at each lag. Similarly, the time-dependent models (sARIMAt) fit independent sARIMA-type models to the hourly-specific data, approximately increasing the number of paramters by a factor 24 compared to the sARIMA and sARIMAX models. The GARCH-ARX model results in the lowest AIC, suggesting that this model provides the best compromise between model complexity and fit. 

The MAE and RSME suggest that the sARIMA model provides the best fit to the data and, interestingly, the same measures suggest that the sARIMAX model has one of the worst. The VAR-type models have the highest RMSE statstics but MAE statistics that comparable to other models, suggesting that these models have more extreme errors. The 

| Model | sARIMA | sARIMAX | sARIMAt | sARIMAXt |  VAR  |  VARX | GARCH-AR | GARCH-ARX | 
|:------|:------:|:-------:|:-------:|:--------:|:-----:|:-----:|:--------:|:---------:| 
| AIC   | -59500 |  -49200 |    1480 |     3830 |  -108 |  -107 |   -66300 |    -68500 |
| MAE   | 0.0776 |   0.143 |   0.125 |    0.127 | 0.129 | 0.126 |    0.105 |     0.101 |
| RMSE  |  0.133 |   0.273 |   0.250 |    0.253 | 0.337 | 0.335 |    0.202 |     0.195 |

Table 1: Summary statistics for each model tested within-sample model fit. The Akaike information criteria (AIC), mean absolute error (MAE) of the fitted to observed values, and the root mean square error (RMSE) of the fitted to observed values.

The predictive power of all models tested is summarised in Table 2, where the out-sample MAE and RMSE statistics are given. From these statistics the sARIMA and sARIMAt models provided the greatest predictive power with approximately the same errors. The VAR model's diverge the most from their within-sample to out-of-sample model fit, possibly indicating over-fitting. This could be attributed to the large number of parameters, which is reflected in the high AIC statistics in Table 1.

| Model | sARIMA | sARIMAX | sARIMAt | sARIMAXt |  VAR  |  VARX | GARCH-AR | GARCH-ARX | 
|:------|:------:|:-------:|:-------:|:--------:|:-----:|:-----:|:--------:|:---------:|
| MAE   |  0.136 |  0.159  |   0.135 |    0.149 | 0.181 | 0.197 |    0.177 |     0.175 |
| RMSE  |  0.185 |  0.220  |   0.186 |    0.203 | 0.240 | 0.258 |    0.246 |     0.243 |

Table 2: Summary statistics for each model tested out-of-sample model fit. The Akaike information criteria (AIC), mean absolute error (MAE) of the fitted to observed values, and the root mean square error (RMSE) of the fitted to observed values.

## Model assumptions

The models used above come with associated assumptions. Mostly, the ARIMA-type and VAR models assume that residuals are independently and identically distributed by a nomarl distribution with a constant variance. This assumption should be inspected to give a fuller picture of model fit. The standardised model residuals of the sARIMA model, which showed the lowest within-sample MAE and RSME, are shown in Figure 4. If the modelling assumptions were held the residuals would show a constant variance, this assumption is clearly violated. The model still captures the trends of the data fairly well, the key issue is that the resulting standard errors are likely wrong, impacting the trustworthy-ness of confidence intervals for parameter values, fitted values and predicted values. The same, non-constant variance issue appears for all sARIMA- and VAR-type models, although the output is omitted for brevity, making their results questionable. 

The GARCH-type models, assume a dynamic variance and it is modelled as such. Inspecting the standardised residuals again, we see results more inline with the assumptions. The residuals show far fewer points with drastically larger than expected values and are more consitent with a standard normal distribution, as would be expected.

<img src="/static/energy_spot_price_statistical_model_comparison/sarima_std_residuals.png" alt="sARIMA residuals" width="1470" height="1166">
<img src="/static/energy_spot_price_statistical_model_comparison/garch_std_residuals.png" alt="GARCH residuals" width="1318" height="1118">


# Conclusion

From the range of statistical models tested the sARIMA with not exogenous variables showed lowest within- and out-of-sample errors, suggesting that this model captured the data the best. However, the data did not conform to one of the ARIMA modelling assumptions, namely, constant variance. Although the model captured the trend in the data well, due to this failure the model's predictive intervals should not be trusted. The issue was seen in the other constant-variance models investigated within this blog. The GARCH-type models, however, did not show the same problem. These models a dyanmic volatility, which is modelled. Although these models did not show as close a fit to the data, assessed by the within- and out-of-sample MAEs and RSMEs, as the sARIMA model, the lower AICs and better diagnostics suggest that this is more appropriate for predicting hourly energy spot prices. 
