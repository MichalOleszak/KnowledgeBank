# Info ------------------------------------------------------------------------
# The forecastxgb package provides time series modelling and forecasting functions 
# that combine the machine learning approach of Chen, He and Benesty's xgboost with 
# the convenient handling of time series and familiar API of Rob Hyndman's forecast. 
# It applies to time series the Extreme Gradient Boosting proposed in Greedy Function 
# Approximation: A Gradient Boosting Machine, by Jermoe Friedman in 2001.


# Settings --------------------------------------------------------------------
# devtools::install_github("ellisp/forecastxgb-r-package/pkg")
library(forecastxgb)
library(fpp)  # for datasets


# The workhorse function is xgbar. This fits a model to a time series. Under the hood, 
# it creates a matrix of explanatory variables based on lagged versions of the response 
# time series, and (optionally) dummy variables of some sort for seasons. That matrix 
# is then fed as the feature set for xgboost to do its stuff.


# Extreme Gradient Boosting AutoRegression model (XGBAR) ----------------------
model <- xgbar(gas)


# By default, xgbar uses row-wise cross-validation to determine the best number of 
# rounds of iterations for the boosting algorithm without overfitting. A final model 
# is then fit on the full available dataset.


# Inspecting importance of features -------------------------------------------
xgbar_importance(model)
summary(model)


# Forecasting wihtout regressors ----------------------------------------------
fc <- forecast(model, h = 12)
plot(fc)

# Forecasting with external regressors ----------------------------------------
consumption <- usconsumption[ ,1]
income <- matrix(usconsumption[ ,2], dimnames = list(NULL, "Income"))
consumption_model <- xgbar(y = consumption, xreg = income)
summary(consumption_model)


# Seasonality -----------------------------------------------------------------
# Currently there are three methods of treating seasonality:
#  - The current default method is to throw dummy variables for each season into 
#    the mix of features for xgboost to work with.
#  - An alternative is to perform classic multiplicative seasonal adjustment on the 
#    series before feeding it to xgboost. This seems to work better.
#  - A third option is to create a set of pairs of Fourier transform variables 
#    and use them as x regressors 
model <- xgbar(gas, seas_method = "dummies")
fc <- forecast(model, h = 12)
plot(fc)

model <- xgbar(gas, seas_method = "decompose")  # probably best
fc <- forecast(model, h = 12)
plot(fc)

model <- xgbar(gas, seas_method = "fourier")
fc <- forecast(model, h = 12)
plot(fc)


# Transformations -------------------------------------------------------------
# The data can be transformed by a modulus power transformation (as per John 
# and Draper, 1980) before feeding to xgboost. This transformation is similar 
# to a Box-Cox transformation, but works with negative data. Leaving the default
# lambda parameter as 1 will effectively switch off this transformation.

# Unsure if playing with lambda is good for this model!


# Non-stationarity ------------------------------------------------------------
# trend_method can be set to auto.arima-style differencing, which is based on 
# successive KPSS tests until there is no significant evidence the remaining 
# series is non-stationary.
model <- xgbar(AirPassengers, trend_method = "none", seas_method = "fourier")
plot(forecast(model, 24))


model <- xgbar(AirPassengers, trend_method = "differencing", seas_method = "fourier")
plot(forecast(model, 24))









