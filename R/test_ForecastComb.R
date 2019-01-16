install.packages('ForecastComb', dependencies = TRUE)
#devtools::install_github("ceweiss/ForecastComb")
library(ForecastComb)

example("auto_combine")

data("electricity")
plot(electricity, plot.type = "single")
lines(electricity[,6], col = "red")

train_actual <- electricity[1:100, 6]
train_pred <- electricity[1:100, 1:5]
test_actual <- electricity[101:123, 6]
test_pred <- electricity[101:123, 1:5]
data <- foreccomb(train_actual, train_pred, test_actual, test_pred)


# Example on simulated data ---------------------------------------------------
# Generate data 
obs <- rnorm(100)
preds <- matrix(rnorm(1000, 1), 100, 10)
train_o <- obs[1:80]
train_p <- preds[1:80,]
test_o <- obs[81:100]
test_p <- preds[81:100,]
data <- foreccomb(train_o, train_p, test_o, test_p)

comb_BG(data)


# Evaluating all the forecast combination methods and returning the best.
# If necessary, it uses the built-in automated parameter optimisation methods
# for the different methods.
best_combination_mape <- auto_combine(data, criterion = "MAPE")
best_combination_rmse <- auto_combine(data, criterion = "RMSE")
best_combination_mae  <- auto_combine(data, criterion = "MAE")

# Same as above, but now we restrict the parameter ntop_pred for the method comb_EIG3 to be 3.
param_list <- list()
param_list$comb_EIG3$ntop_pred <- 3
best_combination_restricted <- auto_combine(data, criterion = "MAPE", param_list = param_list)
