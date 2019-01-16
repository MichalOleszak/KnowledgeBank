# Fitting the model -----------------------------------------------------------
data(mtcars)

# Multiple Linear Regression Example 
fit <- lm(mpg ~ cyl + disp + hp, data = mtcars)
summary(fit)# show results

# Other useful functions 
coefficients(fit) # model coefficients
confint(fit, level = 0.95) # CIs for model parameters 
fitted(fit) # predicted values
residuals(fit) # residuals
anova(fit) # anova table 
vcov(fit) # covariance matrix for model parameters 
influence(fit) # regression diagnostics


# Diagnostics plots -----------------------------------------------------------
# Diagnostic plots provide checks for heteroscedasticity, normality, 
# and influential observerations.
layout(matrix(c(1,2,3,4),2,2)) # optional 4 graphs/page 
plot(fit)


# Comparing models ------------------------------------------------------------
# You can compare nested models with the anova( ) function. The following code 
# provides a simultaneous test that disp and hp add to linear prediction above 
# and beyond cyl.
fit1 <- lm(mpg ~ cyl + disp + hp, data = mtcars)
fit2 <- lm(mpg ~ cyl, data = mtcars)
anova(fit1, fit2)


# Cross-validation ------------------------------------------------------------
# You can do K-Fold cross-validation using the cv.lm() function in the 
# DAAG package.
library(DAAG)
fit <- lm(mpg ~ cyl + disp + hp, data = mtcars)
cv.lm(data = mtcars, fit, m = 3) # 3 fold cross-validation

# Sum the MSE for each fold, divide by the number of observations, and take 
# the square root to get the cross-validated standard error of estimate.

# You can assess R2 shrinkage via K-fold cross-validation. Using the crossval() 
# function from the bootstrap package, do the following:
fit <- lm(mpg ~ cyl + disp + hp, data = mtcars)
library(bootstrap)

# define functions 
theta.fit <- function(x,y){lsfit(x,y)}
theta.predict <- function(fit,x){cbind(1,x)%*%fit$coef} 

# matrix of predictors
X <- as.matrix(mtcars[c("cyl", "disp", "hp")])
# vector of predicted values
y <- as.matrix(mtcars[c("mpg")]) 

results <- crossval(X, y, theta.fit, theta.predict, ngroup=10)
cor(y, fit$fitted.values)**2 # raw R2 
cor(y,results$cv.fit)**2 # cross-validated R2


# Variable selection ----------------------------------------------------------
# Selecting a subset of predictor variables from a larger set (e.g., stepwise 
# selection) is a controversial topic. You can perform stepwise selection 
# (forward, backward, both) using the stepAIC() function from the MASS package. 
# stepAIC() performs stepwise model selection by exact AIC.
library(MASS)
fit <- lm(mpg ~ cyl + disp + hp, data = mtcars)
step <- stepAIC(fit, direction = "both")
step$anova # display results

# Alternatively, you can perform all-subsets regression using the leaps() 
# function from the leaps package. In the following code nbest indicates 
# the number of subsets of each size to report. Here, the ten best models 
# will be reported for each subset size (1 predictor, 2 predictors, etc.).
library(leaps)
leaps <- regsubsets(mpg ~ cyl + disp + hp, data = mtcars, nbest = 10)
# view results 
summary(leaps)
# plot a table of models showing variables in each model.
# models are ordered by the selection statistic.
plot(leaps,scale = "r2")
# plot statistic by subset size 
library(car)
subsets(leaps, statistic = "rsq")
# Other options for plot() are bic, Cp, and adjr2. Other options for 
# plotting with subset() are bic, cp, adjr2, and rss.


# Relative importance ---------------------------------------------------------
# The relaimpo package provides measures of relative importance for each of 
# the predictors in the model. See help(calc.relimp) for details on the four 
# measures of relative importance provided.

# Calculate Relative Importance for Each Predictor
library(relaimpo)
fit <- lm(mpg ~ cyl + disp + hp, data = mtcars)
calc.relimp(fit, type=c("lmg", "last", "first", "pratt"), rela = TRUE)

# Bootstrap Measures of Relative Importance (1000 samples) 
boot <- boot.relimp(fit, b = 1000, type = c("lmg", "last", "first", "pratt"), 
                    rank = TRUE, diff = TRUE, rela = TRUE)
booteval.relimp(boot) # print result
plot(booteval.relimp(boot, sort = TRUE)) # plot result
