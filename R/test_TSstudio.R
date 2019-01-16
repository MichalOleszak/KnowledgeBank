# TSstudio for seasonal analysis of time series and visualisation 
# source: https://cran.r-project.org/web/packages/TSstudio/vignettes/TSstudio_Intro.html

# Get package and data --------------------------------------------------------
install.packages("TSstudio")
library(TSstudio)
library(xts)
library(zoo)
library(quantmod)
# Loading the stock price of key technology companies:
tckrs <- c("GOOGL", "FB", "AAPL", "MSFT")
getSymbols(tckrs, 
           from = "2013-01-01",
           src = "yahoo")
Google <- GOOGL$GOOGL.Close
closing <- cbind(GOOGL$GOOGL.Close, FB$FB.Close, AAPL$AAPL.Close, MSFT$MSFT.Close)
names(closing) <- c("Google", "Facebook", "Apple", "Microsoft")
data(USgas)


# Plot time series ------------------------------------------------------------
ts_plot(USgas,
        title = "US Natural Gas Consumption 2000 - 2017",
        Xtitle = "Source: U.S. Bureau of Transportation Statistics",
        Ytitle = "Billion Cubic Feet",
        slider = TRUE
)

ts_plot(closing, 
        title = "Top Technology Companies Stocks Prices Since 2013",
        type = "single")
ts_plot(closing, 
        title = "Top Technology Companies Stocks Prices Since 2013",
        type = "multiple")


# Seasonality analysis --------------------------------------------------------
ts_seasonal(USgas, type = "normal")
ts_seasonal(USgas, type = "cycle")
ts_seasonal(USgas, type = "box")
ts_seasonal(USgas, type = "all")

ts_heatmap(USgas)
ts_surface(USgas)

# Polar plot: the year is represented by color and the magnitude is represented 
# by the size of the cycle unit layer
ts_polar(USgas)

# Seasonal decomposition
ts_decompose(USgas, type = "both")


# Correlation analysis --------------------------------------------------------
ts_acf(USgas)
ts_pacf(USgas)

ts_lags(USgas, lag.max = 15)


# Split the data into training and testing sets -------------------------------
# (leaving the last 12 months for testing)
split_USgas <- ts_split(USgas, sample.out = 12)
train <- split_USgas$train
test <- split_USgas$test


# Residual analysis -----------------------------------------------------------
library(forecast)
fit <- auto.arima(train, lambda = BoxCox.lambda(train))
check_res(fit)


# Forecast evaluation ---------------------------------------------------------
fc <- forecast(fit, h = 12)
test_forecast(actual = USgas, forecast.obj = fc, test = test)


# Other utilities -------------------------------------------------------------

# ts to data frame 
data("USVSales")
library(DT)
USVSales_df <- ts_reshape(USVSales)
datatable(USVSales_df, filter = 'top', options = list(
  pageLength = nrow(USVSales_df), autoWidth = TRUE
))

# Converting "zoo" or "xts" objects to "ts" class
data("Michigan_CS")
Michigan_CS_ts <- xts_to_ts(Michigan_CS)
