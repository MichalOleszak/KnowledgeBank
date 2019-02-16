# *** Tabular data: caret & iml *** ---------------------------------------------------------------

# data wrangling
library(tidyverse)
library(readr)
# ml
library(caret)
# plotting
library(gridExtra)
library(grid)
library(ggridges)
library(ggthemes)
theme_set(theme_minimal())
# explaining models
devtools::install_github("christophM/iml")
library(iml)
devtools::install_github("pbiecek/breakDown")
library(breakDown)
devtools::install_github("pbiecek/DALEX")
library(DALEX)


# Get data ----------------------------------------------------------------------------------------
wine_data <- read_csv("C:/Users/Michal/Dropbox/R/data/wine_quality_red.csv", 
                      col_types = cols(total.sulfur.dioxide = col_double())) %>%
  mutate(quality = as.factor(ifelse(quality < 6, "qual_low", "qual_high")))

colnames(wine_data) <- gsub("\\.", "_", colnames(wine_data))
glimpse(wine_data)


# EDA ---------------------------------------------------------------------------------------------
p1 <- wine_data %>%
  ggplot(aes(x = quality, fill = quality)) +
  geom_bar(alpha = 0.8) +
  scale_fill_tableau() +
  guides(fill = FALSE)

p2 <- wine_data %>%
  gather(x, y, fixed_acidity:alcohol) %>%
  ggplot(aes(x = y, y = quality, color = quality, fill = quality)) +
  facet_wrap( ~ x, scale = "free", ncol = 3) +
  scale_fill_tableau() +
  scale_color_tableau() +
  geom_density_ridges(alpha = 0.8) +
  guides(fill = FALSE, color = FALSE)

grid.arrange(p1, p2, ncol = 2, widths = c(0.3, 0.7))


# Model -------------------------------------------------------------------------------------------
# Split for training and testing: 80/20
set.seed(42)
idx <- createDataPartition(wine_data$quality, 
                           p = 0.8, 
                           list = FALSE, 
                           times = 1)

wine_train <- wine_data[ idx,]
wine_test  <- wine_data[-idx,]

# Run a RF with 3 times repeated 5-fold cross-validation
fit_control <- trainControl(method = "repeatedcv",
                            number = 5,
                            repeats = 3)

rf_model <- train(quality ~ ., 
                  data = wine_train, 
                  method = "rf", 
                  preProcess = c("scale", "center"),
                  trControl = fit_control,
                  verbose = FALSE)

# Evaluate on test set
test_predict <- predict(rf_model, wine_test)
confusionMatrix(test_predict, as.factor(wine_test$quality))


# Explaining/interpreting the model ---------------------------------------------------------------

# 1. Feature importance ----
#
# The varImp() function from the caret package can be used to calculate feature importance measures for most methods. 
# For Random Forest classification models such as ours, the prediction error rate is calculated for permuted 
# out-of-bag data of each tree and permutations of every feature These two measures are averaged and normalized.
#
# Here are the definitions of the variable importance measures. The first measure is computed from permuting OOB data: 
# For each tree, the prediction error on the out-of-bag portion of the data is recorded (error rate for classification, 
# MSE for regression). Then the same is done after permuting each predictor variable. The difference between the two
# are then averaged over all trees, and normalized by the standard deviation of the differences. If the standard 
# deviation of the differences is equal to 0 for a variable, the division is not done (but the average is almost 
# always equal to 0 in that case). The second measure is the total decrease in node impurities from splitting on 
# the variable, averaged over all trees. For classification, the node impurity is measured by the Gini index. 
# For regression, it is measured by residual sum of squares.

rf_model_imp <- varImp(rf_model, scale = TRUE)
p1 <- rf_model_imp$importance %>%
  as.data.frame() %>%
  rownames_to_column() %>%
  ggplot(aes(x = reorder(rowname, Overall), y = Overall)) +
  geom_bar(stat = "identity", fill = "#1F77B4", alpha = 0.8) +
  coord_flip()

# We can also use a ROC curve for evaluating feature importance. For this, we have the caret::filterVarImp() function.
#
# The importance of each predictor is evaluated individually using a "filter" approach. For classification, ROC curve 
# analysis is conducted on each predictor. For two class problems, a series of cutoffs is applied to the predictor 
# data to predict the class. The sensitivity and specificity are computed for each cutoff and the ROC curve is computed.
# The trapezoidal rule is used to compute the area under the ROC curve. This area is used as the measure of variable 
# importance. For multi-class outcomes, the problem is decomposed into all pair-wise problems and the area under the 
# curve is calculated for each class pair (i.e class 1 vs. class 2, class 2 vs. class 3 etc.). For a specific class, 
# the maximum area under the curve across the relevant pair-wise AUC's is used as the variable importance measure. 
# For regression, the relationship between each predictor and the outcome is evaluated. An argument, nonpara, is used 
# to pick the model fitting technique. When nonpara = FALSE, a linear model is fit and the absolute value of the 
# t-value for the slope of the predictor is used. Otherwise, a loess smoother is fit between the outcome and the 
# predictor. The R^2 statistic is calculated for this model against the intercept only null model.

roc_imp <- filterVarImp(x = wine_train[, -ncol(wine_train)], y = wine_train$quality)
p2 <- roc_imp %>%
  as.data.frame() %>%
  rownames_to_column() %>%
  ggplot(aes(x = reorder(rowname, qual_high), y = qual_high)) +
  geom_bar(stat = "identity", fill = "#1F77B4", alpha = 0.8) +
  coord_flip()

grid.arrange(p1, p2, ncol = 2, widths = c(0.5, 0.5))



# IML package 

# In order to work with iml, we need to adapt our data a bit by removing the response variable 
# and the creating a new predictor object that holds the model, the data and the class labels.
# The iml package uses R6 classes: New objects can be created by calling Predictor$new().

X <- wine_train %>%
  select(-quality) %>%
  as.data.frame()

predictor <- Predictor$new(rf_model, data = X, y = wine_train$quality)
str(predictor)


# 2. Partial Dependence Plots ---------------------------------------------------------------------

# Besides knowing which features were important, we are interested in how the features influence 
# the predicted outcome. The Partial class implements partial dependence plots and individual 
# conditional expectation curves. Each individual line represents the predictions (y-axis) for one
# data point when we change one of the features (e.g. 'lstat' on the x-axis). The highlighted line
# is the point-wise average of the individual lines and equals the partial dependence plot. The marks 
# on the x-axis indicates the distribution of the 'lstat' feature, showing how relevant a region is for 
# interpretation (little or no points mean that we should not over-interpret this region)

pdp_obj <- Partial$new(predictor, feature = "alcohol")
pdp_obj$center(min(wine_train$alcohol))
glimpse(pdp_obj$results)

# The partial dependence plot calculates and plots the dependence of f(X) on a single or two features. 
# It's the aggregate of all individual conditional expectation curves, that describe how, for a single 
# observation, the prediction changes when the feature changes.

pdp_obj$plot()

pdp_obj2 <- Partial$new(predictor, feature = c("sulphates", "pH"))
pdp_obj2$plot()


# 3. Feature interaction --------------------------------------------------------------------------

# Interactions between features are measured via the decomposition of the prediction function: 
# If a feature j has no interaction with any other feature, the prediction function can be expressed
# as the sum of the partial function that depends only on j and the partial function that only depends 
# on features other than j. If the variance of the full function is completely explained by the sum of
# the partial functions, there is no interaction between feature j and the other features. Any variance
# that is not explained can be attributed to the interaction and is used as a measure of interaction 
# strength. The interaction strength between two features is the proportion of the variance of the 
# 2-dimensional partial dependence function that is not explained by the sum of the two 1-dimensional 
# partial dependence functions. The interaction measure takes on values between 0 (no interaction) to 1.

interact <- Interaction$new(predictor, feature = "alcohol")

#plot(interact)
interact$results %>%
  ggplot(aes(x = reorder(.feature, .interaction), y = .interaction, fill = .class)) +
  facet_wrap(~ .class, ncol = 2) +
  geom_bar(stat = "identity", alpha = 0.8) +
  scale_fill_tableau() +
  coord_flip() +
  guides(fill = FALSE)


# 4. Tree Surrogate -------------------------------------------------------------------------------

# The tree surrogate method uses decision trees on the predictions.
# A conditional inference tree is fitted on the predicted from the machine learning model and the data. 
# The partykit package and function are used to fit the tree. By default a tree of maximum depth of 2 
# is fitted to improve interpretability.

tree <- TreeSurrogate$new(predictor, maxdepth = 5)

# The R^2 value gives an estimate of the goodness of fit or how well the decision tree approximates the model.
tree$r.squared

#plot(tree)
tree$results %>%
  mutate(prediction = colnames(select(., .y.hat.qual_high, .y.hat.qual_low))[max.col(select(., .y.hat.qual_high, .y.hat.qual_low),
                                                                                     ties.method = "first")],
         prediction = ifelse(prediction == ".y.hat.qual_low", "qual_low", "qual_high")) %>%
  ggplot(aes(x = prediction, fill = prediction)) +
  facet_wrap(~ .path, ncol = 5) +
  geom_bar(alpha = 0.8) +
  scale_fill_tableau() +
  guides(fill = FALSE)


# 5. LocalModel -----------------------------------------------------------------------------------

# LocalModel is a implementation of the LIME algorithm from Ribeiro et al. 2016, similar to lime.
# According to the LIME principle, we can look at individual predictions. Here, for example on the 
# first row of the test set:
X2 <- wine_test[, -12]
i = 1
lime_explain <- LocalModel$new(predictor, x.interest = X2[i, ])
lime_explain$results

#plot(lime_explain)
p1 <- lime_explain$results %>%
  ggplot(aes(x = reorder(feature.value, -effect), y = effect, fill = .class)) +
  facet_wrap(~ .class, ncol = 2) +
  geom_bar(stat = "identity", alpha = 0.8) +
  scale_fill_tableau() +
  coord_flip() +
  labs(title = paste0("Test case #", i)) +
  guides(fill = FALSE)

# . or for the sixth row:
i = 6
lime_explain$explain(X2[i, ])
p2 <- lime_explain$results %>%
  ggplot(aes(x = reorder(feature.value, -effect), y = effect, fill = .class)) +
  facet_wrap(~ .class, ncol = 2) +
  geom_bar(stat = "identity", alpha = 0.8) +
  scale_fill_tableau() +
  coord_flip() +
  labs(title = paste0("Test case #", i)) +
  guides(fill = FALSE)

grid.arrange(p1, p2, ncol = 2)


# 6. Shapley value  -------------------------------------------------------------------------------

# Shapley computes feature contributions for single predictions with the Shapley value, an approach 
# from cooperative game theory. The features values of an instance cooperate to achieve the prediction. 
# The Shapley value fairly distributes the difference of the instance's prediction and the datasets 
# average prediction among the features.

shapley <- Shapley$new(predictor, x.interest = X2[1, ])
head(shapley$results)

#shapley$plot()
shapley$results %>%
  ggplot(aes(x = reorder(feature.value, -phi), y = phi, fill = class)) +
  facet_wrap(~ class, ncol = 2) +
  geom_bar(stat = "identity", alpha = 0.8) +
  scale_fill_tableau() +
  coord_flip() +
  guides(fill = FALSE)


# 7. breakDown ------------------------------------------------------------------------------------

# Model agnostic tool for decomposition of predictions from black boxes. Break Down Table shows 
# contributions of every variable to a final prediction. Break Down Plot presents variable contributions 
# in a concise graphical way. This package work for binary classifiers and general regression models.
# The broken() function decomposes model predictions and outputs the contributions of each feature to 
# the final prediction.

predict.function <- function(model, new_observation) {
  predict(model, new_observation, type="prob")[,2]
}
predict.function(rf_model, X2[1, ])

br <- broken(model = rf_model, 
             new_observation = X2[1, ], 
             data = X, 
             baseline = "Intercept", 
             predict.function = predict.function, 
             keep_distributions = TRUE)

#plot(br)
data.frame(y = br$contribution,
           x = br$variable) %>%
  ggplot(aes(x = reorder(x, y), y = y)) +
  geom_bar(stat = "identity", fill = "#1F77B4", alpha = 0.8) +
  coord_flip()

# If we set keep_distributions = TRUE, we can plot these distributions of partial predictions, 
# as well as the average.

plot(br, plot_distributions = TRUE)


# 8. DALEX ----------------------------------------------------------------------------------------
# (Descriptive mAchine Learning EXplanations)
# Machine Learning (ML) models are widely used and have various applications in classification or regression. 
# Models created with boosting, bagging, stacking or similar techniques are often used due to their high performance, 
# but such black-box models usually lack of interpretability. DALEX package contains various explainers that help
# to understand the link between input variables and model output. The single_variable() explainer extracts conditional
# response of a model as a function of a single selected variable. It is a wrapper over packages 'pdp' and 'ALEPlot'. 
# The single_prediction() explainer attributes parts of a model prediction to particular variables used in the model. 
# It is a wrapper over 'breakDown' package. The variable_dropout() explainer calculates variable importance scores 
# based on variable shuffling. All these explainers can be plotted with generic plot() function and compared across 
# different models.

# Create an explain object
p_fun <- function(object, newdata) {
  predict(object, newdata = newdata, type = "prob")[, 2]
}
yTest <- as.numeric(wine_test$quality)
explainer_classif_rf <- DALEX::explain(rf_model, 
                                       label = "rf",
                                       data = wine_test, 
                                       y = yTest,
                                       predict_function = p_fun)

# Analyze model performance as the distribution of residuals
mp_classif_rf <- model_performance(explainer_classif_rf)
plot(mp_classif_rf)
plot(mp_classif_rf, geom = "boxplot")

# Feature importance as the loss from variable dropout
vi_classif_rf <- variable_importance(explainer_classif_rf, loss_function = loss_root_mean_square)
plot(vi_classif_rf)

# Marginal response for a single variable
# As type we can choose between 'pdp' for Partial Dependence Plots and 'ale' for Accumulated Local Effects.
pdp_classif_rf  <- variable_response(explainer_classif_rf, variable = "alcohol", type = "pdp")
plot(pdp_classif_rf)

ale_classif_rf  <- variable_response(explainer_classif_rf, variable = "alcohol", type = "ale")
plot(ale_classif_rf)
