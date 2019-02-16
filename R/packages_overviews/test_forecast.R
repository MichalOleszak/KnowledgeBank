# ***** TS ANALYSIS ***** ----

library(forecast)
library(fpp2)
library(ggplot2)
library(seasonal)
library(hts) # hierachical time series
library(tidyverse)

# Plotting and testing ---------------------------------------------------------
autoplot(a10)

# Seasonal patterns
ggseasonplot(a10)
ggseasonplot(a10, polar = TRUE)
beer <- window(ausbeer, 1992)
ggsubseriesplot(beer)

# Autocorrelation
autoplot(oil)
gglagplot(oil)
ggAcf(oil) # within blue lines = white noise

# Ljung-Box test
# The Ljung-Box test considers the first h autocorrelation values together. 
# A significant test (small p-value) indicates the data are probably not white noise.
Box.test(diff(goog), lag = 10, type = "Ljung")


# Calendar adjustments --------------------------------------------------------
# For monthly data, there will be variation between the months simply because of 
# the different numbers of days in each month. Solution: remove it!
dframe <- cbind(Monthly = milk,
                DailyAverage = milk / monthdays(milk))
autoplot(dframe, facet = TRUE) +
  xlab("Years") + ylab("Pounds") +
  ggtitle("Milk production per cow")


# Residual analysis -----------------------------------------------------------
# Critical assumptions:
#   - Residuals are uncorrelated (otherwise, there is info in them not caught by the model)
#   - Residuals have mean zero (trivial: if not, adjust foreacast with the bias)
# Assumptions needed for confidence intervals:
#   - Residudals have constant variance
#   - Residuals are normally distributed
#
# The Ljing-Box test (or Breusch-Godfrey test for regression) is a Lagrange Multiplier test for serial correlation. 
# It is used to test the joint hypothesis that there is no autocorrelation in the residuals up to a certain specified order. 
# A small p-value indicates there is significant autocorrelation remaining in the residuals.
goog %>% naive() %>% checkresiduals()
ausbeer %>% snaive() %>% checkresiduals()


# Prediction intervals --------------------------------------------------------
# y_(t+h) = c * sigma_h, where c is the qunatile of the errors distribution 
# (e.g. 1.96 for 95% interval assuming errors normality) and sigma_h is the estimate 
# of the std of the forecast distribution at time h. For one step ahead, residuals' std 
# is a good estimate. For multiple steps ahead, residuals have to be assumed uncorrelated.
# Then sigma_h equals (sigma denotes residuals' std):
#  - mean forecast: sigma * sqrt(1 + 1/T)
#  - naive forecast: sigma * sqrt(h)
#  - seasonal naive forecast: sigma * sqrt(k+1), where k is integer part of (h-1)/m
#  - drift forecast: sigma * sqrt(h * (1 + h/T))
# Note that when h = 1 and T is large, these all give the same approximate value of sigma.

# Prediction intervals from bootstrapped residuals -> when errors normality unreasonable
# y_(T+1) = y_forecast(T+1|T) + error sampled from residuals
# y_(T+2) = y_forecast(T+1|T+1) + error sampled from residuals
# ...

# If a transformation has been used, then the prediction interval should be computed on the 
# transformed scale, and the end points back-transformed to give a prediction interval on the 
# original scale. This approach preserves the probability coverage of the prediction interval, 
# although it will no longer be symmetric around the point forecast.

# Prediction intervals for multiple regression:
# X - predictors matrix
# y - dependent variable vector
# xf - row vector with future predictor values for forecasting
# y_hat = xf * Beta_hat = xf(X'X)^-1(X'y)
# estimated forecast variance: sigma^2[1 + xf(X'X)^-1(xf)'] where sigma^2 is residual variance
# as estimated with: [(y-X*beta_hat)'(y-X*beta_hat)] / (T-k-1)
# 95% interval:
# y_hat +- 1.96 * sigma^2 *sqrt(1 + xf(X'X)^-1(xf)')


# Forecast accuracy -----------------------------------------------------------
# - A forecast method that minimizes the MAE will lead to forecasts of the median, 
#   while minimizing the RMSE will lead to forecasts of the mean.
# - Percentage errors assume the unit of measurement has a meaningful zero.
train <- subset(gold, end = 1000)
naive_fc <- naive(train, h = 108)
mean_fc  <- meanf(train, h = 108)

accuracy(naive_fc, gold)
accuracy(mean_fc, gold)

# Expanding window
# Compute cross-validated errors for up to 8 steps ahead
e <- matrix(NA_real_, nrow = 1000, ncol = 8)
for (h in 1:8) {
  e[, h] <- tsCV(goog, forecastfunction = naive, h = h)
}
# Compute the MSE values and remove missing values
mse <- colMeans(e^2, na.rm = TRUE)
# Plot the MSE values against the forecast horizon
data.frame(h = 1:8, MSE = mse) %>%
  ggplot(aes(x = h, y = MSE)) + geom_point()


# Box-Cox transformations  ----------------------------------------------------
# Box-Cox trasformations for variance stabilization:
#   w_t = log(y_t) if lambda == 0
#   w_t = (y_t^lambda - 1) / lambda otherwise 
# lambda = 1 -> no substantive transformation
# lambda = 0.5 -> sqrt + linear transformation
# lambda = 0.33 -> cube root + linear transformation
# lambda = 0 -> natural log transformation
# lambda = -1 -> inverse transformation
# Transformations sometimes make little difference to the forecasts but have
# a large effect on prediction intervals.

autoplot(a10)
a10 %>% BoxCox(lambda = 0) %>% autoplot()
a10 %>% BoxCox(lambda = 0.1) %>% autoplot()
a10 %>% BoxCox(lambda = 0.2) %>% autoplot()
a10 %>% BoxCox(lambda = 0.3) %>% autoplot()

# The back-transformed forecast will not be the mean of the forecast distribution, but often the median 
# (when that the distribution on the transformed space is symmetric). The larger the forecast variance, 
# the bigger the difference between the mean and the median. We have to correct for this bias.
fc <- rwf(eggs, drift=TRUE, lambda=0, h=50, level=80)
fc2 <- rwf(eggs, drift=TRUE, lambda=0, h=50, level=80,
           biasadj=TRUE)
autoplot(eggs) +
  autolayer(fc, series="Simple back transformation") +
  autolayer(fc2, series="Bias adjusted", PI=FALSE) +
  guides(colour=guide_legend(title="Forecast"))

# Automatically optimized lambda value:
BoxCox.lambda(a10)

# Non-seasonal differencing for stationarity
autoplot(wmurders)
autoplot(diff(wmurders))

# Seasonal differencing for stationarity
# With seasonal data, differences are taken between observations in the same 
# season of consecutive years. This is called seasonal differencing.
# Sometimes both seasonal differences and lag-1 differences should be applied 
# to the same series, thus, calculating the differences in the differences.
autoplot(h02)
difflogh02 <- diff(log(h02), lag = 12)
autoplot(difflogh02)
ddifflogh02 <- diff(difflogh02)
autoplot(ddifflogh02)


# Stationarity ----------------------------------------------------------------
# Making the data stationary
# When both seasonal and first differences are applied, it makes no difference which is 
# done first-the result will be the same. However, if the data have a strong seasonal pattern, 
# we recommend that seasonal differencing be done first, because the resulting series will 
# sometimes be stationary and there will be no need for a further first difference. If first 
# differencing is done first, there will still be seasonality present.
cbind("Billion kWh" = usmelec,
      "Logs" = log(usmelec),
      "Seasonally\n differenced logs" = diff(log(usmelec),12),
      "Doubly\n differenced logs" = diff(diff(log(usmelec),12),1)) %>%
  autoplot(facets=TRUE) +
  xlab("Year") + ylab("") +
  ggtitle("Monthly US net electricity generation")

# Unit root tests
tseries::adf.test(AirPassengers, alternative = "stationary")
tseries::kpss.test(AirPassengers)



# ***** METHODS ***** ----

# Basic methods ---------------------------------------------------------------
# Naive forecast (last value)
fcgoog <- naive(goog, 20)
autoplot(fcgoog)
summary(fcgoog)

# Seasonal naive forecast (last value of corresponding season)
fcbeer <- snaive(ausbeer, 16)
autoplot(fcbeer)
summary(fcbeer)

# Mean forecast
fcgoog2 <- meanf(diff(goog), 20)
autoplot(fcgoog2)
summary(fcgoog2)


# Linear regression -----------------------------------------------------------
fit <- tslm(Consumption ~ trend + season + Income + Production + Unemployment + Savings, data = uschange)
summary(fit)
# Variable selection
CV(fit)


# Natural Cubic Smoothing Splines ---------------------------------------------
# (impose constraints so the spline function is linear at the end)
marathon %>%
  splinef(lambda = 0) %>% 
  autoplot()


# Time Series Decomposition ---------------------------------------------------
# Estimate Trend-Cycle component with moveing average
# (larger MA order means a smoother curve)
autoplot(elecsales, series="Data") +
  autolayer(ma(elecsales, order = 5), series = "5-MA") +
  xlab("Year") + ylab("GWh") +
  ggtitle("Annual electricity sales: South Australia") +
  scale_colour_manual(values = c("Data" = "grey50", "5-MA" = "red"),
                      breaks = c("Data", "5-MA"))

autoplot(elecequip, series="Data") +
  autolayer(ma(elecequip, 12), series="12-MA") +
  xlab("Year") + ylab("New orders index") +
  ggtitle("Electrical equipment manufacturing (Euro area)") +
  scale_colour_manual(values=c("Data"="grey","12-MA"="red"),
                      breaks=c("Data","12-MA"))

# Classical decomposition
# - the estimate of the trend-cycle is unavailable for the first few and last few observations
# - the trend-cycle estimate tends to over-smooth rapid rises and falls in the data
# - classical decomposition methods assume that the seasonal component repeats from year to year
elecequip %>% decompose(type="multiplicative") %>%
  autoplot() + xlab("Year") +
  ggtitle("Classical multiplicative decomposition
    of electrical equipment index")

# X11 decomposition
# This method is based on classical decomposition, but includes many extra steps and features 
# in order to overcome its drawbacks. In particular, trend-cycle estimates are available for 
# all observations including the end points, and the seasonal component is allowed to vary slowly 
# over time. X11 also has some sophisticated methods for handling trading day variation, holiday 
# effects and the effects of known predictors. It handles both additive and multiplicative decomposition. 
# The process is entirely automatic and tends to be highly robust to outliers and level shifts in the time series.
elecequip %>% seas(x11 = "") -> fit
autoplot(fit) +
  ggtitle("X11 decomposition of electrical equipment index")

autoplot(elecequip, series="Data") +
  autolayer(trendcycle(fit), series="Trend") +
  autolayer(seasadj(fit), series="Seasonally Adjusted") +
  xlab("Year") + ylab("New orders index") +
  ggtitle("Electrical equipment manufacturing (Euro area)") +
  scale_colour_manual(values=c("gray","blue","red"),
                      breaks=c("Data","Seasonally Adjusted","Trend"))

fit %>% seasonal() %>% ggsubseriesplot() + ylab("Seasonal")


# SEATS decomposition (Seasonal Extraction in ARIMA Time Series)
# The procedure works only with quarterly and monthly data. So seasonality of other kinds, 
# such as daily data, or hourly data, or weekly data, require an alternative approach.
elecequip %>% seas() %>%
  autoplot() +
  ggtitle("SEATS decomposition of electrical equipment index")


# STL decomposition (Seasonal and Trend decomposition using Loess)
# - Unlike SEATS and X11, STL will handle any type of seasonality, not only monthly and quarterly data
# - The seasonal component is allowed to change over time, and the rate of change can be controlled by the user
# - The smoothness of the trend-cycle can also be controlled by the user.
# - It can be robust to outliers (i.e., the user can specify a robust decomposition), so that occasional unusual 
#   observations will not affect the estimates of the trend-cycle and seasonal components. They will, however, 
#   affect the remainder component.
# - STL does not handle trading day or calendar variation automatically, and it only provides facilities for additive decompositions
elecequip %>%
  stl(t.window = 13, s.window = "periodic", robust = TRUE) %>%
  autoplot()
# The two main parameters to be chosen when using STL are the trend-cycle window (t.window) and the seasonal window (s.window). 
# These control how rapidly the trend-cycle and seasonal components can change. Smaller values allow for more rapid changes. 
# Both t.window and s.window should be odd numbers and refer to the number of consecutive years to be used when estimating the 
# trend-cycle and seasonal components respectively. The user must specify s.window as there is no default. Setting it to be infinite 
# is equivalent to forcing the seasonal component to be periodic (i.e., identical across years). Specifying t.window is optional, 
# and a default value will be used if it is omitted.


# Forecasting with decomposition
# Approach: predict trend-cycle (any method) and sesonality (seasonal naive method) separately
fit <- stl(elecequip, t.window=13, s.window="periodic", robust=TRUE)
fit %>% seasadj() %>% naive() %>%
  autoplot() + ylab("New orders index") +
  ggtitle("Naive forecasts of seasonally adjusted data")
fit %>% forecast(method="naive") %>%
  autoplot() + ylab("New orders index")
# stlf() will decompose the time series using STL, forecast the seasonally adjusted series, and return reseasonalize the forecasts
stlf(elecequip, method = 'naive') %>% autoplot()
stlf(elecequip, method = 'ets') %>% autoplot()


# Exponential smoothing -------------------------------------------------------
# SES (Simple Exponential Smoothing) - same value for all horizons
fc <- ses(marathon, h = 10)
summary(fc)
autoplot(fc) + autolayer(fitted(fc))

# Holt model - linear trend (can be damped = vanish with time)
fc1 <- holt(austa, h = 30, PI = FALSE)
fc2 <- holt(austa, damped = TRUE, h = 30, PI = FALSE)
autoplot(austa) + xlab("Year") + ylab("value") +
  autolayer(fc1, series = "Linear trend") +
  autolayer(fc2, series = "Damped trend")

# Holt-Winters model - trend & seasonality
# If seasonal variation increases with time - use multiplicative method
fc <- hw(a10, seasonal = "multiplicative", h = 36)
autoplot(fc)

# Holt-Winters with daily data
fc <- hw(subset(hyndsight,end=length(hyndsight)-35),
         damped = TRUE, seasonal="multiplicative", h=35)
autoplot(hyndsight) +
  autolayer(fc, series="HW multi damped", PI=FALSE)+
  guides(colour=guide_legend(title="Daily forecasts"))

# State space exponential smoothing models
# 3 trend possibilities: none, additive, dumped additive
# 3 seasonality possibilities: none, additive, multiplicative
# 2 possible error specifications: additive, multiplicative (noise increases with series' level)
# 3*3*2=18 possible exponential smoothing methods!
# ets() [error, trend, seasonality] estimates parametes by MLE and then chooses best ETS 
# combination (out of the 18) by minimizing AIC_c (bias corrected AIC)
fiths <- ets(hyndsight)
checkresiduals(fiths)
autoplot(forecast(fiths))


# ARIMA models ----------------------------------------------------------------
# Choosing p and q
# - ACF plot shows the autocorrelations which measure the relationship between y_t and y_t-k
#   for different values of k.
# If y_t and y_t-1 are correlated, then y_t-1 and y_t-2 must also be. Then, y_t and y_t-2 might be
# correlated because they are related to y_t-1, not beacuse y_t-2 has new info helpful in forecasting y_t.
# - PACF plot measures the realtionship between y_t and y_t-k after removing the effect
#   of lags 1, 2, 3, ..., k-1
# The 1st PACF = the 1st ACF, as there is nothing between them to remove
# Each partial autocorrelation can be estimated as the last coefficient in an autoregressive model,
# i.e. PACF_k = phi_k in AR(k)
ggtsdisplay(diff(AirPassengers))
# These plots can only be used for ARIMA(p,d,0) and ARIMA(0,d,q)! Not if both q and p are positive!
# Data follow ARIMA(p, d, 0) if in the DIFFERENCED data plots:
#  - the ACF is exponentially decaying or sinusoidal
#  - there is a significant spike at lag p in the PACF, but none beyond lag p
# Data follow ARIMA(0, d, q) if in the DIFFERENCED data plots:
#  - the PACF is exponentially decaying or sinusoidal
#  - there is a significant spike at lag q in the ACF, but none beyond lag q 

# Determine the number of first-diffs or seasonal differences needed, based on ADF and KPSS
ndiffs(AirPassengers)
nsdiffs(AirPassengers)

# auto.arima() selects d using unit root test and estimates p and q using ML,
# minimizing AICc; drift parameter is the constat in the model
auto.arima(austa) %>% forecast(h = 10) %>% autoplot()

# Manually specified arima
usnetelec %>%
  Arima(order = c(2,1,2), include.constant = TRUE) %>%
  forecast() %>%
  autoplot()

# Seasonal ARIMA 
auto.arima(h02, lambda = 0) %>% forecast(24) %>% autoplot()
# The seasonal part of an AR or MA model will be seen in the seasonal lags of the PACF and ACF. 
# ARIMA(0,0,0)(0,0,1)[12] model will show:
#   - a spike at lag 12 in the ACF but no other significant spikes;
#   - exponential decay in the seasonal lags of the PACF (i.e., at lags 12, 24, 36, .).
# ARIMA(0,0,0)(1,0,0)[12] model will show:
#   - exponential decay in the seasonal lags of the ACF;
#   - a single significant spike at lag 12 in the PACF.

# Look at more models to choose best (slow!)
# (lower AICc is better)
auto.arima(euretail)$aicc
auto.arima(euretail, stepwise = FALSE, approximation = FALSE)$aicc

# How does auto.arima work? Hyndman-Khandakar algorithm for automatic ARIMA modelling:
# 1. d is determined using repeated KPSS tests
# 2. p and q ara chosen by minimizing AICc:
#  a) four or five initial models are estimated (depending on the values of d)
#  b) best model from a) is set to be the current model
#  c) variations of the current model are considered:
#       -> p, q +- 1
#       -> include/exclude constant
#     if a better model found, it is set to be the current model
# 3. Repeat 2c) until no better model can be found

# Unit roots
# Stationarity condition: p complex root of 1 - phi_1*B^1 - ... - phi_p*B^p lie outside the unit circle
# Invertibility condition: q complex root of 1 + theta_1*B^1 + ... + theta_p*B^p lie outside the unit circle
# It is easier to plot the inverse roots instead, as they should all lie within the unit circle.
elecequip %>% stl(s.window='periodic') %>% seasadj() -> eeadj
fit <- Arima(eeadj, order=c(3,1,1))
autoplot(fit)
# Any roots close to the unit circle may be numerically unstable, and the corresponding model will not be good for forecasting.
# The Arima() function will never return a model with inverse roots outside the unit circle. The auto.arima() function 
# is even stricter, and will not select a model with roots close to the unit circle either.


# Dynamic regression ----------------------------------------------------------
# Regression model in which error term follows an arima process (in OLS it is white noise)
# Estimates obtained by minimizing squared residuals in the arima error equation
# All variable should be stationary, if not -> take differences
auto.arima(advert[, "sales"], xreg = advert[, "advert"], stationary = TRUE) %>%
  forecast(xreg = rep(10, 6)) %>%
  autoplot()

# Two types of errors: regression equation and error equation
fit <- auto.arima(uschange[,"Consumption"], xreg=uschange[,"Income"])
cbind("Regression Errors" = residuals(fit, type="regression"),
      "ARIMA errors" = residuals(fit, type="innovation")) %>%
  autoplot(facets=TRUE)
# It is the ARIMA errors that should resemble a white noise series.
checkresiduals(fit)

# Quadratic Dynamic Regression
ggplot(as.data.frame(elecdaily), aes(Temperature, Demand)) + geom_point()
autoplot(elecdaily[, c("Demand", "Temperature")], facets = TRUE)

xreg <- cbind(MaxTemp = elecdaily[, "Temperature"], 
              MaxTempSq = elecdaily[, "Temperature"]^2, 
              Workday = elecdaily[, "WorkDay"])
fit <- auto.arima(elecdaily[, "Demand"], xreg = xreg)
# Forecast fit one day ahead for a working day with temp 20 degrees
forecast(fit, xreg = cbind(20, 400, 1))

# Stochastic and Deterministic trend
# y_t = beta_0 + beta_1 * t + eta_t
# Deterministic: eta_t is an ARMA process
# Stochastic: eta_t is an ARIMA proces with d=1, which means that
#             y_t = y_t-1 + beta_1 * t + eta*_t, where eta*_t is an ARMA process
trend <- seq_along(austa)
(fit1 <- auto.arima(austa, d=0, xreg=trend)) # deterministic trend
(fit2 <- auto.arima(austa, d=1))             # stochastic trend
fc1 <- forecast(fit1, xreg=data.frame(trend=length(austa)+1:10))
fc2 <- forecast(fit2, h=10)
autoplot(austa) +
  autolayer(fc2, series="Stochastic trend") +
  autolayer(fc1, series="Deterministic trend") +
  ggtitle("Forecasts from deterministic and stochastic trend models") +
  xlab("Year") + ylab("Visitors to Australia (millions)") +
  guides(colour=guide_legend(title="Forecast"))
# Point estimates are similar, but the prediction intervals are not. Stochastic trends 
# have much wider prediction intervals because the errors are non-stationary.


# Dynamic harmonic regression -------------------------------------------------
# Uses Fourier terms to handle seasonality -> series of sin and cos functions
# of right frequency can approximate any periodic function.
# Good when there are long seasonal periods.
# Errors follow a non-seasonal arima process (seasonality captured by Fourier).
# Fourier assumes seasonal pattern does not change over time (unlike SARIMA).
# K determines complexity of the seasonal pattern (larger = more complex).
# One should select K as to minimize AICc.
fit <- auto.arima(auscafe, xreg = fourier(auscafe, K = 5), seasonal = FALSE, lambda = 0)
fit %>%
  forecast(xreg = fourier(auscafe, K = 5, h = 24)) %>%
  autoplot()

# Forecasting weekly data
harmonics <- fourier(gasoline, K = 13)
  # forecasts next 3 years
newharmonics <- fourier(gasoline, K = 13, h = 52 * 3)
auto.arima(gasoline, xreg = harmonics, seasonal = FALSE) %>%
  forecast(fit, xreg = newharmonics) %>% 
  autoplot()

# Choosing K
cafe04 <- window(auscafe, start=2004)
plots <- list()
for (i in seq(6)) {
  fit <- auto.arima(cafe04, xreg = fourier(cafe04, K = i),
                    seasonal = FALSE, lambda = 0)
  plots[[i]] <- autoplot(forecast(fit,xreg=fourier(cafe04, K=i, h=24))) +
    xlab(paste("K=",i,"   AICC=",round(fit$aicc,2))) +
    ylab("") + ylim(1.5,4.7)
}
gridExtra::grid.arrange(plots[[1]],plots[[2]],plots[[3]],
                        plots[[4]],plots[[5]],plots[[6]], nrow=3)

# Harmonic regression for multiple seasonality
# taylor contains half-hourly electricity demand in England and Wales over a few months
# in the year 2000. The seasonal periods are 48 (daily seasonality) and 7 x 48 = 336 
# (weekly seasonality).
# auto.arima would be slow for such a long series: use tslm instead

# Fit a harmonic regression using order 10 for each type of seasonality
fit <- tslm(taylor ~ fourier(taylor, K = c(10, 10)))
# Forecast 20 working days ahead
forecast(fit, newdata = data.frame(fourier(taylor, K = c(10, 10), h = 24 * 2 * 20))) %>%
  autoplot()

# calls contains 20 consecutive days of 5-minute call volume data for a large 
# North American bank. There are 169 5-minute periods in a working day, and so the 
# weekly seasonal frequency is 5 x 169 = 845 and no weekly seasonality, just daily.
autoplot(calls)
xreg <- fourier(calls, K = c(10, 0))
fit <- auto.arima(calls, xreg = xreg, seasonal = F, stationary = T)
# Plot forecasts for 10 working days ahead
forecast(fit, xreg =  fourier(calls, c(10, 0), h = 10 * 169)) %>%
  autoplot()


# TBATS models ----------------------------------------------------------------
# - Trigonometric terms for seasonality (can change over time)
# - Box-Cox transformation for heterogeneity
# - ARMA errors for short-term dynamics
# - Trend (possibly damped)
# - Seasonal (including multiple and non-integer perdiods)
# (The model is very slow and prediction intervals are often too wide!)
#
# Notation:
# TBATS(1, {0,0}, -, {<51.18,14>})
# - BoxCox lambda = 1
# - Error = ARMA(0,0)
# - no damping
# - Seasonal period = 51.18 (weekly), 14 Fourier terms
#
# Uses a combination of Fourier terms with an exponential smoothing state space model 
# and a Box-Cox transformation, in a completely automated manner. As with any automated 
# modelling framework, there may be cases where it gives poor results!
#
# A TBATS model differs from dynamic harmonic regression in that the seasonality is allowed 
# to change slowly over time in a TBATS model, while harmonic regression terms force the 
# seasonal patterns to repeat periodically without changing.
autoplot(gas)
fit <- tbats(gas)
fc <- forecast(fit, 60)
autoplot(fc)


# Complex seasonality ---------------------------------------------------------
# msts class handles multiple seasonality and non-integer frequencies
# The mstl() function is a variation on stl() designed to deal with multiple seasonality. 
# It will return multiple seasonal components, as well as a trend and remainder component.
calls %>% forecast::mstl() %>%
  autoplot() + xlab("Week")
# 169 - there are 169 5-minute intervals per day (data comes in 5min intervals)
# 845 - weekly seasonal patterns: 169*5
# The trend and the weekly seasonality have relatively narrow ranges (Y axes) compared to the other 
# components, because there is very little trend seen in the data, and the weekly seasonality is weak.

# STL with multiple seasonal periods
# The decomposition can also be used in forecasting, with each of the seasonal components 
# forecast using a seasonal naïve method, and the seasonally adjusted data forecasting using ETS
calls %>%  stlf() %>%
  autoplot() + xlab("Week")

# Dynamic harmonic regression with multiple seasonal periods
# Because there are multiple seasonalities, we need to add Fourier terms for each seasonal period.
fit <- auto.arima(calls, seasonal=FALSE, lambda=0,
                  xreg=fourier(calls, K=c(10,10)))
fit %>%
  forecast(xreg=fourier(calls, K=c(10,10), h=2*169)) %>%
  autoplot(include=5*169) +
  ylab("Call volume") + xlab("Weeks")
# This is a very large model, containing 43 parameters: 7 ARMA coefficients, 20 Fourier coefficients 
# for frequency 169, and 16 Fourier coefficients for frequency 845. We don't use all the Fourier terms 
# for frequency 845 because there is some overlap with the terms of frequency 169 (since  845=5*169).


# VAR models ------------------------------------------------------------------
# A VAR model is a generalisation of the univariate autoregressive model for forecasting a vector 
# of time series.19 It comprises one equation per variable in the system. The right hand side of each 
# equation includes a constant and lags of all of the variables in the system.
#
# If the series are stationary, we forecast them by fitting a VAR to the data directly (known as a "VAR in levels"). 
# If the series are non-stationary, we take differences of the data in order to make them stationary, then fit a VAR model 
# (known as a "VAR in differences"). In both cases, the models are estimated equation by equation using the principle of least squares.
#
# Another possibility is that the series may be non-stationary but cointegrated, which means that there exists 
# a linear combination of them that is stationary. In this case, a VAR specification that includes an error correction 
# mechanism (usually referred to as a vector error correction model) should be included, and alternative estimation 
# methods to least squares estimation should be used.
#
# There are two decisions one has to make when using a VAR to forecast, namely how many variables (K) 
# and how many lags (p) should be included in the system. The number of coefficients to be estimated in a VAR is equal to
# K + pK^2, or 1 + pK per equation. The more coefficients that need to be estimated, the larger the estimation error entering the forecast.
# K is usually kept small (only variables that are correlated), p chosen with information criteria (BIC).
library(vars, quietly=TRUE, warn.conflicts=FALSE)
VARselect(uschange[,1:2], lag.max=8, type="const") # based on BIC (SC) we choose VAR(1) and check also VAR(2)
var1 <- VAR(uschange[,1:2], p=1, type="const")
var2 <- VAR(uschange[,1:2], p=2, type="const")
serial.test(var1, lags.pt=10, type="PT.asymptotic")
serial.test(var2, lags.pt=10, type="PT.asymptotic")
# Both a VAR(1) and a VAR(2) have some residual serial correlation (Portmanteau test), and therefore we fit a VAR(3).
var3 <- VAR(uschange[,1:2], p=3, type="const")
serial.test(var3, lags.pt=10, type="PT.asymptotic")
# The residuals for this model pass the test for serial correlation.
forecast(var3) %>%
  autoplot() + xlab("Year")


# NNAR models -----------------------------------------------------------------
# (Neural Network Autoregression)
# Artificial neural networks are forecasting methods that are based on simple mathematical models of the brain. 
# They allow complex nonlinear relationships between the response variable and its predictors.
#
# Feed-forward networks with one hidden layer: NNAR(p,k), with p lagged inputs and k nodes in the hidden layer
# NNAR(p, 0) is therefore equvalent to ARIMA(p,0,0), but without the restrictions on the parameters to ensure stationarity.
#
# With seasonal data, it is useful to also add the last observed values from the same season as inputs.
# NNAR(p,P,k)_m has inputs: y_t-1, y_t-2, ..., y_t-p, y_t-m, t_t-2m, ...,  y_t-Pm and k neurons in the hidden layer.
# NNAR(p,P,0)_m is equvalent to ARIMA(p,0,0)(P,0,0)_m
# NNAR(3,1,2)_12 has inputs y_t-1, y_t-2, y_t-3, y_t-12, and 2 neurons in the hidden layer. 
fit <- nnetar(sunspotarea, lambda=0)
autoplot(forecast(fit,h=30))

# Prediction intervals: simulate future paths by resampling errors
sim <- ts(matrix(0, nrow=30L, ncol=9L),
          start=end(sunspotarea)[1L]+1L)
for(i in seq(9)) {
  sim[,i] <- simulate(fit, nsim=30L)
}
autoplot(sunspotarea) + autolayer(sim)
# If we do this a few hundred or thousand times, we can get a very good picture of the forecast distributions. 
# This is how the forecast() function produces prediction intervals for NNAR models.
fcast <- forecast(fit, PI=TRUE, h=30)
autoplot(fcast)


# Bootstrapping & Bagging -----------------------------------------------------
# Bootstrapping time series
#
# First, the time series is Box-Cox-transformed, and then decomposed into trend, seasonal and remainder 
# components using STL. Then we obtain shuffled versions of the remainder component to get bootstrapped
# remainder series. Because there may be autocorrelation present in an STL remainder series we use a block bootstrap.
bootseries <- bld.mbb.bootstrap(debitcards, 10) %>%
  as.data.frame() %>% ts(start=2000, frequency=12)
autoplot(debitcards) +
  autolayer(bootseries, colour=TRUE) +
  autolayer(debitcards, colour=FALSE) +
  ylab("Bootstrapped series") + guides(colour="none")
# This type of bootstrapping can be useful in two ways:
#  - it helps us to get a better measure of forecast uncertainty;
#  - it provides a way of improving our point forecasts using bagging.

# Prediction intervals from bootstrapped series
#
# Almost all prediction intervals from time series models are too narrow. This is a well-known phenomenon 
# and arises because they do not account for all sources of uncertainty.
# There are at least four sources of uncertainty in forecasting using time series models:
#  - The random error term;
#  - The parameter estimates;
#  - The choice of model for the historical data;
#  - The continuation of the historical data generating process into the future.
# We can use bootstrapped time series to go some way towards overcoming this problem.
#
# First, we simulate many time series that are similar to the original data
nsim <- 1000L
sim <- bld.mbb.bootstrap(debitcards, nsim)
# For each of these series, we fit an ETS model and simulate one sample path from that model.
# A different ETS model may be selected in each case, although it will most likely select the same model because 
# the series are very similar. However, the estimated parameters will be different. Therefore the simulated sample 
# paths will allow for model uncertainty and parameter uncertainty, as well as the uncertainty associated with the 
# random error term.
h <- 36L
future <- matrix(0, nrow=nsim, ncol=h)
for(i in seq(nsim)) {
  future[i,] <- simulate(ets(sim[[i]]), nsim=h)
}
# Finally, we take the means and quantiles of these simulated sample paths to form point forecasts and prediction intervals.
start <- tsp(debitcards)[2]+1/12
simfc <- structure(list(
  mean = ts(colMeans(future), start=start, frequency=12),
  lower = ts(apply(future, 2, quantile, prob=0.025),
             start=start, frequency=12),
  upper = ts(apply(future, 2, quantile, prob=0.975),
             start=start, frequency=12),
  level=95),
  class="forecast")
# These prediction intervals will be larger than those obtained from an ETS model applied directly to the original data.
etsfc <- forecast(ets(debitcards), h=h, level=95)
autoplot(debitcards) +
  ggtitle("Monthly retail debit card usage in Iceland") +
  xlab("Year") + ylab("million ISK") +
  autolayer(simfc, series="Simulated") +
  autolayer(etsfc, series="ETS")
  
# Bagged ETS forecasts 
#
# Another use for these bootstrapped time series is to improve forecast accuracy. If we produce forecasts from 
# each of the additional time series, and average the resulting forecasts, we get better forecasts than if we simply 
# forecast the original time series directly. This is called "bagging" which stands for "bootstrap aggregating".
start <- tsp(debitcards)[2]+1/12
sim <- bld.mbb.bootstrap(debitcards, 10) %>%
  as.data.frame() %>%
  ts(frequency=12, start=2000)
fc <- purrr::map(as.list(sim),
                 function(x){forecast(ets(x))[["mean"]]}) %>%
  as.data.frame() %>%
  ts(frequency=12, start=start)
autoplot(debitcards) +
  autolayer(sim, colour=TRUE) +
  autolayer(fc, colour=TRUE) +
  autolayer(debitcards, colour=FALSE) +
  ylab("Bootstrapped series") +
  guides(colour="none")
# The average of these forecasts gives the bagged forecasts of the original data.
etsfc <- debitcards %>% ets() %>% forecast(h=36)
baggedfc <- debitcards %>% baggedETS() %>% forecast(h=36)
autoplot(debitcards) +
  autolayer(baggedfc, series="BaggedETS", PI=FALSE) +
  autolayer(etsfc, series="ETS", PI=FALSE) +
  guides(colour=guide_legend(title="Forecasts"))


# Hierarchical or grouped series ----------------------------------------------
# Hierarchical time series
# Example: number of turists in Australia nationalwide, by state and by zone within each state
#
# To create a hierarchical time series, we use the hts() function. The function requires 
# two inputs: the bottom-level time series and information about the hierarchical structure. 
#
# There are several ways to input the structure of the hierarchy. In this case we are using 
# the characters argument. The first three characters of each column name of visnights capture 
# the categories at the first level of the hierarchy (States). The following five characters 
# capture the bottom-level categories (Zones).
tourism.hts <- hts(visnights, characters = c(3, 5))

# Plot by state
tourism.hts %>% aggts(levels=0:1) %>%
  autoplot(facet=TRUE) +
  xlab("Year") + ylab("millions") + ggtitle("Visitor nights")

# Plot by zone
cols <- sample(scales::hue_pal(h=c(15,375),
                               c=100,l=65,h.start=0,direction = 1)(NCOL(visnights)))
as_tibble(visnights) %>%
  gather(Zone) %>%
  mutate(Date = rep(time(visnights), NCOL(visnights)),
         State = str_sub(Zone,1,3)) %>%
  ggplot(aes(x=Date, y=value, group=Zone, color=Zone)) +
  geom_line() +
  facet_grid(State~., scales="free_y") +
  xlab("Year") + ylab("millions") +
  ggtitle("Visitor nights by Zone") +
  scale_color_manual(values = cols)
  
# Grouped time series
# 
# With grouped time series, the structure does not naturally disaggregate in a unique hierarchical manner, 
# and often the disaggregating factors are both nested and crossed. For example, we could further disaggregate 
# all geographical levels of the Australian tourism data by purpose of travel (such as holidays, business, etc.). 
# So we could consider visitors nights split by purpose of travel for the whole of Australia, and for each state, 
# and for each zone. Then we describe the structure as involving the purpose of travel "crossed" with the 
# geographical hierarchy.
#
# Example: Australian prison population by state, legal status and gender. The three are crossed, but none
# are nested within the others.
#
# To create a grouped time series, we use the gts() function.
prison.gts <- gts(prison/1e3, characters = c(3,1,9),
                  gnames = c("State", "Gender", "Legal",
                             "State*Gender", "State*Legal", "State*Gender*Legal"))

# Basic plot
prison.gts %>% aggts(level=0:3) %>% autoplot()

# Fancy plot
p1 <- prison.gts %>% aggts(level=0) %>%
  autoplot() + ggtitle("Australian prison population") +
  xlab("Year") + ylab("Total number of prisoners ('000)")
groups <- aggts(prison.gts, level=1:3)
cols <- sample(scales::hue_pal(h=c(15,375),
                               c=100,l=65,h.start=0,direction = 1)(NCOL(groups)))
p2 <- as_tibble(groups) %>%
  gather(Series) %>%
  mutate(Date = rep(time(groups), NCOL(groups)),
         Group = str_extract(Series, "([A-Za-z ]*)")) %>%
  ggplot(aes(x=Date, y=value, group=Series, color=Series)) +
  geom_line() +
  xlab("Year") + ylab("Number of prisoners ('000)") +
  scale_color_manual(values = cols) +
  facet_grid(.~Group, scales="free_y") +
  scale_x_continuous(breaks=seq(2006,2016,by=2)) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
gridExtra::grid.arrange(p1, p2, ncol=1)

# The bottom-up approach
#
# This approach involves first generating forecasts for each series at the bottom-level, and then 
# summing these to produce forecasts for all the series in the structure.
#
# Forecasts can be produced using the forecast() function applied to objects created by hts() or gts(). 
# The hts package has three in-built options to produce forecasts: ETS models, ARIMA models or random walks; 
# these are controlled by the fmethod argument. It also use several methods for producing coherent forecasts, 
# controlled by the method argument.
forecast(prison.gts, method="bu", fmethod="arima")

# Top-down approaches
#
# They only work with strictly hierarchical aggregation structures, and not with grouped structures. 
# They involve first generating forecasts for the Total series and then disaggregating these down the hierarchy.
# For this we need the disaggregation proportions.
#  - Average historical proportions: Each proportion reflects the average of the historical proportions of the bottom-level series
#    relative to the total aggregate. This approach is implemented in the forecast() function by setting method="tdgsa", where 
#    tdgsa stands for "top-down Gross-Sohl method A".
#  - Proportions of the historical averages: Each proportion captures the average historical value of the bottom-level series
#    relative to the average value of the total aggregate. This approach is implemented in the forecast() function by setting 
#    method="tdgsf", where tdgsa stands for "top-down Gross-Sohl method F".

# Forecast proportions
#
# Because historical proportions used for disaggregation do not take account of how those proportions may change over time, 
# top-down approaches based on historical proportions tend to produce less accurate forecasts at lower levels of the hierarchy 
# than bottom-up approaches. To address this issue, proportions based on forecasts rather than historical data can be used.
#
# This approach is implemented in the forecast() function by setting method="tdfp", where tdfp stands for "top-down forecast proportions".

# Middle-out appraoch
#
# The middle-out approach combines bottom-up and top-down approaches. First, a "middle level" is chosen and forecasts are generated 
# for all the series at this level. For the series above the middle level, coherent forecasts are generated using the bottom-up 
# approach by aggregating the "middle-level" forecasts upwards. For the series below the "middle level", coherent forecasts are 
# generated using a top-down approach by disaggregating the "middle level" forecasts downwards.
#
# This approach is implemented in the forecast() function by setting method="mo" and by specifying the appropriate middle level 
# via the level argument. For the top-down disaggregation below the middle level, the top-down forecast proportions method is used.

# The optimal reconciliation approach
#
# Optimal forecast reconciliation will occur if we can find the P matrix which minimises the forecast error of the set of coherent forecasts.
#
# Using the default arguments for the forecast() function, we compute coherent forecasts for the Australian prison population by 
# the optimal reconciliation approach with the WLS estimator using variance scaling.
prisonfc <- forecast(prison.gts)
fcsts <- aggts(prisonfc, levels=0:3)
groups <- aggts(prison.gts, levels=0:3)
autoplot(fcsts) + autolayer(groups)

prisonfc <- ts(rbind(groups, fcsts),
               start=start(groups), frequency=4)
p1 <- autoplot(prisonfc[,"Total"]) +
  ggtitle("Australian prison population") +
  xlab("Year") + ylab("Total number of prisoners ('000)") +
  geom_vline(xintercept=2017)
cols <- sample(scales::hue_pal(h=c(15,375),
                               c=100,l=65,h.start=0,direction = 1)(NCOL(groups)))
p2 <- as_tibble(prisonfc[,-1]) %>%
  gather(Series) %>%
  mutate(Date = rep(time(prisonfc), NCOL(prisonfc)-1),
         Group = str_extract(Series, "([A-Za-z ]*)")) %>%
  ggplot(aes(x=Date, y=value, group=Series, color=Series)) +
  geom_line() +
  xlab("Year") + ylab("Number of prisoners ('000)") +
  scale_color_manual(values = cols) +
  facet_grid(. ~ Group, scales="free_y") +
  scale_x_continuous(breaks=seq(2006,2018,by=2)) +
  theme(axis.text.x = element_text(angle=90, hjust=1)) +
  geom_vline(xintercept=2017)
gridExtra::grid.arrange(p1, p2, ncol=1)


# ***** PRACTICAL ISSUES ***** ----

# Time series of counts -------------------------------------------------------
# Croston method (no parameter estimation, assuming alpha = 0.1 by default)
productC %>% croston() %>% autoplot()


# Ensuring forecasts stay within limits ---------------------------------------
# Positive forecasts: set Box-Cox lambda to 0 (log transform)
eggs %>%
  ets(model="AAN", damped=FALSE, lambda=0) %>%
  forecast(h=50, biasadj=TRUE) %>%
  autoplot()
# Because we set biasadj=TRUE, the forecasts are the means of the forecast distributions.

# Forecasts constrained to an interval (a, b)
# y = log((x-a)/(b-x)), where y is transformed and x original data
# reverse transformation is: x = (((b-a)e^y)/(1 + e^y)) + a
a <- 50
b <- 400
# Transform data and fit model
fit <- log((eggs-a)/(b-eggs)) %>%
  ets(model="AAN", damped=FALSE)
fc <- forecast(fit, h=50)
# Back-transform forecasts
fc[["mean"]] <- (b-a)*exp(fc[["mean"]]) /
  (1+exp(fc[["mean"]])) + a
fc[["lower"]] <- (b-a)*exp(fc[["lower"]]) /
  (1+exp(fc[["lower"]])) + a
fc[["upper"]] <- (b-a)*exp(fc[["upper"]]) /
  (1+exp(fc[["upper"]])) + a
fc[["x"]] <- eggs
# Plot result on original scale
autoplot(fc)
# No bias-adjustment has been used here, so the forecasts are the medians of the future distributions. 
# The prediction intervals from these transformations have the same coverage probability as on the transformed 
# scale, because quantiles are preserved under monotonically increasing transformations.
# The prediction intervals lie above 50 due to the transformation. As a result of this artificial (and unrealistic) constraint, 
# the forecast distributions have become extremely skewed.


# Forecast combinations -------------------------------------------------------
# Simple average often hard to beat
train <- window(auscafe, end=c(2012,9))
h <- length(auscafe) - length(train)
ETS <- forecast(ets(train), h=h)
ARIMA <- forecast(auto.arima(train, lambda=0, biasadj=TRUE), h=h)
STL <- stlf(train, lambda=0, h=h, biasadj=TRUE)
NNAR <- forecast(nnetar(train), h=h)
TBATS <- forecast(tbats(train, biasadj=TRUE), h=h)
Combination <- (ETS[["mean"]] + ARIMA[["mean"]] +
                  STL[["mean"]] + NNAR[["mean"]] + TBATS[["mean"]])/5
autoplot(auscafe) +
  autolayer(ETS, series="ETS", PI=FALSE) +
  autolayer(ARIMA, series="ARIMA", PI=FALSE) +
  autolayer(STL, series="STL", PI=FALSE) +
  autolayer(NNAR, series="NNAR", PI=FALSE) +
  autolayer(TBATS, series="TBATS", PI=FALSE) +
  autolayer(Combination, series="Combination") +
  xlab("Year") + ylab("$ billion") +
  ggtitle("Australian monthly expenditure on eating out")

c(ETS = accuracy(ETS, auscafe)["Test set","RMSE"],
  ARIMA = accuracy(ARIMA, auscafe)["Test set","RMSE"],
  `STL-ETS` = accuracy(STL, auscafe)["Test set","RMSE"],
  NNAR = accuracy(NNAR, auscafe)["Test set","RMSE"],
  TBATS = accuracy(TBATS, auscafe)["Test set","RMSE"],
  Combination =
    accuracy(Combination, auscafe)["Test set","RMSE"])


# Prediction intervals for aggregates -----------------------------------------
#
# A common problem is to forecast the aggregate of several time periods of data, using a model 
# fitted to the disaggregated data. For example, we may have monthly data but wish to forecast 
# the total for the next year. Or we may have weekly data, and want to forecast the total for the 
# next four weeks.
#
# If the point forecasts are means, then adding them up will give a good estimate of the total. 
# But prediction intervals are more tricky due to the correlations between forecast errors.
#
# A general solution is to use simulations. Here is an example using ETS models applied to 
# Australian monthly gas production data, assuming we wish to forecast the aggregate gas demand 
# in the next six months.

# First fit a model to the data
fit <- ets(gas/1000)
# Forecast six months ahead
fc <- forecast(fit, h=6)
# Simulate 10000 future sample paths
nsim <- 10000
h <- 6
sim <- numeric(nsim)
for(i in seq_len(nsim)) {
  sim[i] <- sum(simulate(fit, future=TRUE, nsim=h))
}
meanagg <- mean(sim)

# The mean of the simulations is very close to the sum of the individual forecasts:
sum(fc[["mean"]][1:6])
meanagg

# Prediction intervals are also easy to obtain:
quantile(sim, prob=c(0.1, 0.9))
quantile(sim, prob=c(0.025, 0.975))


# Backcasting -----------------------------------------------------------------
# Function to reverse time
reverse_ts <- function(y) {
  ts(rev(y), start=tsp(y)[1L], frequency=frequency(y))
}
# Function to reverse a forecast
reverse_forecast <- function(object) {
  h <- length(object[["mean"]])
  f <- frequency(object[["mean"]])
  object[["x"]] <- reverse_ts(object[["x"]])
  object[["mean"]] <- ts(rev(object[["mean"]]),
                         end=tsp(object[["x"]])[1L]-1/f, frequency=f)
  object[["lower"]] <- object[["lower"]][h:1L,]
  object[["upper"]] <- object[["upper"]][h:1L,]
  return(object)
}

# Then we can apply these functions to backcast any time series. Here is an example applied to 
# quarterly retail trade in the Euro area. The data are from 1996-2011. We backcast to predict 
# the years 1994-1995.
euretail %>%
  reverse_ts() %>%
  auto.arima() %>%
  forecast() %>%
  reverse_forecast() -> bc
autoplot(bc) +
  ggtitle(paste("Backcasts from",bc[["method"]]))


# Very long and very short time series ----------------------------------------
#
# The sample size required increases with the number of parameters to be estimated, 
# and the amount of noise in the data.
# 
# If data is too short to keep a testing set for out of sample accuracy, the AICc is 
# particularly useful, because it is a proxy for the one-step forecast out-of-sample MSE.
# Choosing the model with the minimum AICc value allows both the number of parameters 
# and the amount of noise to be taken into account.
# 
# Most time series models do not work well for very long time series. The problem is that 
# real data do not come from the models we use. When the number of observations is not large 
# (say up to about 200) the models often work well as an approximation to whatever process 
# generated the data. But eventually we will have enough data that the difference between 
# the true process and the model starts to become more obvious. An additional problem is that 
# the optimization of the parameters becomes more time consuming because of the number of 
# observations involved.
#
# What to do about these issues depends on the purpose of the model. A more flexible and complicated 
# model could be used, but this still assumes that the model structure will work over the whole period 
# of the data. A better approach is usually to allow the model itself to change over time. ETS models 
# are designed to handle this situation by allowing the trend and seasonal terms to evolve over time. 
# ARIMA models with differencing have a similar property. But dynamic regression models do not allow 
# any evolution of model components.
#
# If we are only interested in forecasting the next few observations, one simple approach is to throw 
# away the earliest observations and only fit a model to the most recent observations. Then an inflexible 
# model can work well because there is not enough time for the relationships to change substantially.


# Missing and outlier values --------------------------------------------------
# na.interp() replaces NAs in a time series with estimates
gold2 <- na.interp(gold)
autoplot(gold2, series="Interpolated") +
  autolayer(gold, series="Original") +
  scale_color_manual(
    values=c(`Interpolated`="red",`Original`="gray"))

# If we are willing to assume that the outliers are genuinely errors, or that they won't occur in the 
# forecasting period, then replacing them can make the forecasting task easier. The tsoutliers() function 
# is designed to identify outliers, and to suggest potential replacement values. 
tsoutliers(gold)
gold[768:772]
# Most likely, this was a transcription error, and the correct value should have been $493.70.

# Another useful function is tsclean() which identifies and replaces outliers, and also replaces missing values.
gold %>%
  tsclean() %>%
  ets() %>%
  forecast(h=50) %>%
  autoplot()
