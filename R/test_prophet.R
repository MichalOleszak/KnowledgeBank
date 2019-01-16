# Prophet for foreacsting
# source: https://facebook.github.io/prophet/

# Data prep -------------------------------------------------------------------
setwd("~/Veneficus/Knowledge Sessions/Prophet")
library(prophet)
takeaway_data = read.csv('dataset_takeaway_amsterdam.csv')
takeaway_prophet  = data.frame(y = takeaway_data$n_orders[],
                               ds = as.Date(takeaway_data$orderdate,
                                            '%m/%d/%Y'))
takeaway_monthly <- takeaway_prophet %>%
  group_by(yr = year(ds), mon = month(ds)) %>%
  summarise(y = sum(y)) %>%
  mutate(ds = as.yearmon(paste(yr, mon), '%Y %m')) %>%
  ungroup() %>%
  select(-yr, -mon) %>%
  filter(ds != 'Sep 2017')


# Run model -------------------------------------------------------------------
model_prophet <- prophet(head(takeaway_prophet, - 365))
future        <- make_future_dataframe(model_prophet, periods = 365)
forecast      <- predict(model_prophet, future)


# See forecast ----------------------------------------------------------------
tail(forecast)
prophet_plot_components(model_prophet, forecast)
plot(model_prophet, forecast)


# Saturating forecasts with defined carrying capacity -------------------------
data_sat     <- takeaway_prophet
data_sat$cap <- 10000
# for minimum use floor instead of cap
model_sat  <- prophet(head(data_sat, - 365), growth = 'logistic')
future_sat <- make_future_dataframe(model_sat, periods = 365)
future_sat$cap <- 10000
forecast_sat <- predict(model_sat, future_sat)
plot(model_sat, forecast_sat)


# Adjusting trend flexibility -------------------------------------------------
model_prophet <- prophet(head(takeaway_prophet, - 365), changepoint.prior.scale = 0.5)
# defaultly changepoint.prior.scale = 0.05; larger value makes trend more flexible
future        <- make_future_dataframe(model_prophet, periods = 365)
forecast      <- predict(model_prophet, future)
plot(model_prophet, forecast)


# Specifying changepoint locations --------------------------------------------
model_prophet <- prophet(head(takeaway_prophet, - 365), changepoints = c('2016-07-31'))
future        <- make_future_dataframe(model_prophet, periods = 365)
forecast      <- predict(model_prophet, future)
plot(model_prophet, forecast)


# Modelling holidays / special events -----------------------------------------
xmas <- data_frame(
  holiday = 'xmas',
  ds      = as.Date(c('2012-12-25', '2013-12-25', '2014-12-25', '2015-12-25',
                      '2016-12-25', '2017-12-25', '2018-12-25')),
  lower_window = -1, # include christmas eve
  upper_window = 1   # include boxing day
)
newyear <- data_frame(
  holiday = 'newyear',
  ds      = as.Date(c('2013-01-01', '2014-01-01', '2015-01-01',
                      '2016-01-01', '2017-01-01', '2018-01-01'))
)
holidays <- bind_rows(xmas, newyear)
# check plot to see how holidays are accounted for
model_prophet <- prophet(head(takeaway_prophet, - 365), holidays = holidays, holidays.prior.scale = 100)
# defaultly holidays.prior.scale = 10; smaller value dampens holiday effect
future        <- make_future_dataframe(model_prophet, periods = 365)
forecast      <- predict(model_prophet, future)
plot(model_prophet, forecast)
# check holiday effect
prophet_plot_components(model_prophet, forecast)
forecast %>% 
  select(ds, xmas, newyear) %>% 
  filter(abs(xmas + newyear) > 0) %>%
  tail(10)


# Adding regressors -----------------------------------------------------------
data_expvar  <- data.frame(y = takeaway_data$n_orders,
                          ds = takeaway_data$orderdate,
                          rain = takeaway_data$rain_flag)
data_expvar_in  <- head(data_expvar, (nrow(data_expvar)-365))
data_expvar_fc  <- select(data_expvar, ds, rain)
model_expvar    <- prophet()
model_expvar    <- add_regressor(model_expvar, 'rain')
model_expvar    <- fit.prophet(model_expvar, data_expvar_in)
forecast_expvar <- predict(model_expvar, data_expvar_fc)
plot(model_expvar, forecast_expvar)
prophet_plot_components(model_expvar, forecast_expvar)


# Setting width of predcition bouds (trend uncertainty) -----------------------
model_prophet <- prophet(head(takeaway_prophet, - 365), interval.width = 0.95)
# defaultly interval.width = 0.8


# Adding uncertainty in seasonality -------------------------------------------
# Defaultly only the ones in trend and noise are returned; adding uncertainty
# in seasonality requires full Bayesian sampling - takes same time
model_prophet <- prophet(head(takeaway_prophet, - 365), mcmc.samples = 300)
future        <- make_future_dataframe(model_prophet, periods = 365)
forecast      <- predict(model_prophet, future)
prophet_plot_components(model_prophet, forecast)
# get raw posterior predictive samples
predictive_samples(model_prophet, future)


# Monthly data ----------------------------------------------------------------
model_monthly <- prophet(takeaway_monthly)
future        <- make_future_dataframe(model_monthly, periods = 12, freq = 'month')
forecast      <- predict(model_monthly, future)
plot(model_monthly, forecast)


# Diagnostics -----------------------------------------------------------------
# cross-validated out-of-sample predictions at various cut-offs; takes some time 
model_prophet <- prophet(takeaway_prophet)
cv_pred <- cross_validation(model_prophet, horizon = 10, period = 500, units = 'days')
head(cv_pred)

plot(cv_pred$y[1:10], type = 'l', xlim = c(1,40), ylim = c(2000, 10000))
lines(cv_pred$y[11:20], x = c(11:20))
lines(cv_pred$y[21:30], x = c(21:30))
lines(cv_pred$y[31:40], x = c(31:40))
lines(cv_pred$yhat[1:10], x = c(1:10), col = 'blue')
lines(cv_pred$yhat[11:20], x = c(11:20), col = 'blue')
lines(cv_pred$yhat[21:30], x = c(21:30), col = 'blue')
lines(cv_pred$yhat[31:40], x = c(31:40), col = 'blue')
abline(v = 10.5, col = 'red')
abline(v = 20.5, col = 'red')
abline(v = 30.5, col = 'red')
