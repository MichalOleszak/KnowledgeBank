# CARET - Classification And REgression Training

# Get packages
#install.packages(c('caret', 'skimr', 'RANN', 'randomForest', 'fastAdaboost', 
#                   'gbm', 'xgboost', 'caretEnsemble', 'C50', 'earth'))
# Load the caret package
library(caret)
# Import dataset
orange <- read.csv('https://raw.githubusercontent.com/selva86/datasets/master/orange_juice_withmissing.csv')
# Structure of the dataframe
str(orange)
# See top 6 rows and 10 columns
head(orange[, 1:10])


# *** Data Preparation and Preprocessing *** ------------------------------------------------------

# Create the training and test datasets -----------------------------------------------------------
set.seed(100)

# Step 1: Get row numbers for the training data
# createDataPartition() preserves the proportion of the categories in Y variable
trainRowNumbers <- createDataPartition(orange$Purchase, p = 0.8, list = FALSE)

# Step 2: Create the training  dataset
trainData <- orange[trainRowNumbers,]

# Step 3: Create the test dataset
testData <- orange[-trainRowNumbers,]

# Store X and Y for later use.
x = trainData[, 2:18]
y = trainData$Purchase


# Descriptive statistics --------------------------------------------------------------------------
library(skimr)
skimmed <- skim_to_wide(trainData)
skimmed[, c(1:5, 9:11, 13, 15:16)]


# Impute missings ---------------------------------------------------------------------------------
# Create the knn imputation model on the training data
preProcess_missingdata_model <- preProcess(trainData, method = 'knnImpute')
preProcess_missingdata_model

# Use the imputation model to predict the values of missing data points
library(RANN)  # required for knnInpute
trainData <- predict(preProcess_missingdata_model, newdata = trainData)
anyNA(trainData)


# One-Hot encode dummies --------------------------------------------------------------------------
dummies_model <- dummyVars(Purchase ~ ., data = trainData)

# Create the dummy variables using predict. The Y variable (Purchase) will not be present in trainData_mat.
trainData_mat <- predict(dummies_model, newdata = trainData)

# Convert to dataframe
trainData <- data.frame(trainData_mat)

# See the structure of the new dataset
str(trainData)


# Variable transformations ------------------------------------------------------------------------
#
# Caret offers:
#  - range: Normalize values so it ranges between 0 and 1
#  - center: Subtract Mean
#  - scale: Divide by standard deviation
#  - BoxCox: Remove skewness leading to normality. Values must be > 0
#  - YeoJohnson: Like BoxCox, but works for negative values.
#  - expoTrans: Exponential transformation, works for negative values.
#  - pca: Replace with principal components
#  - ica: Replace with independent components
#  - spatialSign: Project the data to a unit circle

# Convert all the numeric variables to range between 0 and 1
preProcess_range_model <- preProcess(trainData, method = 'range')
trainData <- predict(preProcess_range_model, newdata = trainData)

# Append the Y variable
trainData$Purchase <- y

# Check
apply(trainData[, 1:10], 2, function(x) {c('min' = min(x), 'max' = max(x))})


# *** Visualize the importance of variables *** ---------------------------------------------------

# If you group the X variable by the categories of Y, a significant mean shift amongst the X's groups 
# is a strong indicator that X will have a significant role to help predict Y.
# It is possible to watch this shift visually using box plots and density plots.

featurePlot(x = trainData[, 1:18], 
            y = trainData$Purchase, 
            plot = "box",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")))

# Consider LoyalCHs subplot, which measures the loyalty score of the customer to the CH brand. 
# The mean and the placement of the two boxes are glaringly different. Just by seeing that, we can
# be pretty sure, LoyalCH is going to be a significant predictor of Y.

# Similarly, we can look at density plots.

featurePlot(x = trainData[, 1:18], 
            y = trainData$Purchase, 
            plot = "density",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")))

# In this case, For a variable to be important, I would expect the density curves to be significantly 
# different for the 2 classes, both in terms of the height (kurtosis) and placement (skewness).

# Having visualised the relationships between X and Y, Wwe can only say which variables are likely to 
# be important to predict Y. It may not be wise to conclude which variables are NOT important.
# Because sometimes, variables with uninteresting pattern can help explain certain aspects of Y that 
# the visually important variables may not.


# *** Feature Selection using RFE *** -------------------------------------------------------------

# Feature selection using recursive feature elimination -------------------------------------------
# 
# RFE works in 3 broad steps:
# Step 1: Build a ML model on a training dataset and estimate the feature importances on the test dataset.
# Step 2: Keeping priority to the most important variables, iterate through by building models of given 
#         subset sizes, that is, subgroups of most important predictors determined from step 1. 
#         Ranking of the predictors is recalculated in each iteration.
# Step 3: The model performances are compared across different subset sizes to arrive at the optimal number 
#         and list of final predictors.

# A model to be used:
# lmFuncs - linear regression
# rfFunc - random forests
# nbFuncs - naive Bayes
# treebagFuncs - bagged trees
# caretFuncs -  functions that can be used with caret's train function 

subsets <- c(1:5, 10, 15, 18)
             
ctrl <- rfeControl(functions = rfFuncs,
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

lmProfile <- rfe(x = trainData[, 1:18], 
                 y = trainData$Purchase,
                 sizes = subsets,
                 rfeControl = ctrl)

lmProfile

# The result above is not a mandate that only including these 3 variables will always give high accuracy 
# over larger sized models! That's because the rfe() we just implemented is particular to random forest 
# based rfFuncs.
# Since ML algorithms have their own way of learning the relationship between the x and y, it is not wise 
# to neglect the other predictors, especially when there is evidence that there is information contained in 
# rest of the variables to explain the relationship between x and y.
# Plus also, since the training dataset isn't large enough, the other predictors may not have had the chance 
# to show its worth.


# *** Training and Tuning the model *** -----------------------------------------------------------

# Train model -------------------------------------------------------------------------------------

# See available algorithms in caret
(modelnames <- paste(names(getModelInfo()), collapse=',  '))

# Train a Multivariate Adaptive Regression Splines (MARS)
modelLookup('earth')

# Train the model using MARS and predict on the training data itself.
model_mars = train(Purchase ~ ., data = trainData, method = 'earth')
fitted <- predict(model_mars)

# What train does, apart from building the model:
# - Cross validating the model
# - Tune the hyper parameters for optimal model performance
# - Choose the optimal model based on a given evaluation metric
# - Preprocess the predictors (what we did so far using preProcess())

model_mars
plot(model_mars, main = "Model Accuracies with MARS")


# Compute variable importance ---------------------------------------------------------------------
varimp_mars <- varImp(model_mars)
plot(varimp_mars, main = "Variable Importance with MARS")


# Prepare the test dataset and predict ------------------------------------------------------------

# Now in order to use the model to predict on new data, the data has to be preprocessed and transformed 
# just the way we did on the training data. All the information required for pre-processing is stored in 
# the respective preProcess model and dummyVar model.

# Step 1: Impute missing values 
testData2 <- predict(preProcess_missingdata_model, testData)  

# Step 2: Create one-hot encodings (dummy variables)
testData3 <- predict(dummies_model, testData2)

# Step 3: Transform the features to range between 0 and 1
testData4 <- predict(preProcess_range_model, testData3)

# View
head(testData4[, 1:10])

# Predict on testData
predicted <- predict(model_mars, testData4)
head(predicted)


# Confusion Matrix --------------------------------------------------------------------------------
# Compute the confusion matrix
confusionMatrix(reference = testData$Purchase, data = predicted, mode = 'everything', positive = 'MM')


# *** Hyperparameter tuning *** ----

# Building trainControl ---------------------------------------------------------------------------

# There are two main ways to do hyper parameter tuning using the train():
# 1. Set the tuneLength:
#    tuneLength corresponds to the number of unique values for the tuning parameters caret will 
#    consider while forming the hyper parameter combinations.
# 2. Define and set the tuneGrid:
#    tuneGrid allows to explicitly control what values should be considered for each parameter

# Inside trainControl() you can control how the train() will:
#  3. Cross validation method to use.
#  4. How the results should be summarised using a summary function

# 3. Cross validation method can be one amongst:
#  - 'boot': Bootstrap sampling
#  - 'boot632': Bootstrap sampling with 63.2% bias correction applied
#  - 'optimism_boot': The optimism bootstrap estimator
#  - 'boot_all': All boot methods.
#  - 'cv': k-Fold cross validation
#  - 'repeatedcv': Repeated k-Fold cross validation
#  - 'oob': Out of Bag cross validation
#  - 'LOOCV': Leave one out cross validation
#  - 'LGOCV': Leave group out cross validation

# 4. The summaryFunction can be:
#  - twoClassSummary if Y is binary class
#  - multiClassSummary if the Y has more than 2 categories

# 5. By settiung the classProbs=T the probability scores are generated instead of directly 
#    predicting the class based on a predetermined cutoff of 0.5

fitControl <- trainControl(
  method = 'cv',                   # k-fold cross validation
  number = 5,                      # number of folds
  savePredictions = 'final',       # saves predictions for optimal tuning parameter
  classProbs = T,                  # should class probabilities be returned
  summaryFunction=twoClassSummary  # results summary function
) 

# Tuning with tuneLength --------------------------------------------------------------------------
# Step 1: Tune hyper parameters by setting tuneLength
set.seed(100)
model_mars2 = train(Purchase ~ ., 
                    data = trainData, 
                    method = 'earth', 
                    tuneLength = 5, 
                    metric = 'ROC', 
                    trControl = fitControl)
model_mars2

# Step 2: Predict on testData and Compute the confusion matrix
predicted2 <- predict(model_mars2, testData4)
confusionMatrix(reference = testData$Purchase, 
                data = predicted2, 
                mode = 'everything', 
                positive = 'MM')

# Tuning with tuneGrid -----------------------------------------------------------------------------
# Step 1: Define the tuneGrid
marsGrid <-  expand.grid(nprune = c(2, 4, 6, 8, 10), 
                         degree = c(1, 2, 3))

# Step 2: Tune hyper parameters by setting tuneGrid
set.seed(100)
model_mars3 = train(Purchase ~ ., 
                    data = trainData, 
                    method = 'earth',
                    metric = 'ROC', 
                    tuneGrid = marsGrid, 
                    trControl = fitControl)
model_mars3

# Step 3: Predict on testData and Compute the confusion matrix
predicted3 <- predict(model_mars3, testData4)
confusionMatrix(reference = testData$Purchase, 
                data = predicted3, 
                mode = 'everything', 
                positive = 'MM')


# *** Evalurate multiple algorithms *** ----

# Caret provides the resamples() function where you can provide multiple machine learning 
# models and collectively evaluate them.

set.seed(100)

# Train some models
model_adaboost = train(Purchase ~ ., 
                       data = trainData, 
                       method = 'adaboost', 
                       tuneLength = 2, 
                       trControl = fitControl)

model_rf = train(Purchase ~ ., 
                 data = trainData,
                 method = 'rf', 
                 tuneLength = 5, 
                 trControl = fitControl)

model_xgbDART = train(Purchase ~ ., 
                      data = trainData, 
                      method = 'xgbDART', 
                      tuneLength = 5, 
                      trControl = fitControl, 
                      verbose = F)

model_svmRadial = train(Purchase ~ ., 
                        data = trainData, 
                        method = 'svmRadial', 
                        tuneLength = 15, 
                        trControl = fitControl)

# Compare model performances using resample()
models_compare <- resamples(list(ADABOOST = model_adaboost, 
                                 RF = model_rf, 
                                 XGBDART = model_xgbDART, 
                                 MARS = model_mars3, 
                                 SVM = model_svmRadial))

# Summary of the models performances
summary(models_compare)

# Draw box plots to compare models
scales <- list(x = list(relation = "free"), y = list(relation = "free"))
bwplot(models_compare, scales = scales)


# *** Ensembling the predictions *** ----
library(caretEnsemble)

# Many models in one call -------------------------------------------------------------------------

# Stacking Algorithms - Run multiple algos in one call.
trainControl <- trainControl(method = "repeatedcv", 
                             number = 10, 
                             repeats = 3,
                             savePredictions = TRUE, 
                             classProbs = TRUE)

algorithmList <- c('rf', 'adaboost', 'earth', 'xgbDART', 'svmRadial')

set.seed(100)
models <- caretList(Purchase ~ ., 
                    data = trainData, 
                    trControl = trainControl, 
                    methodList = algorithmList)

results <- resamples(models)
summary(results)

# Box plots to compare models
scales <- list(x = list(relation = "free"), y = list(relation = "free"))
bwplot(results, scales = scales)

# Ensembling models -------------------------------------------------------------------------------

# Make sure you don't use the same trainControl you used to build the models!
set.seed(101)
stackControl <- trainControl(method = "repeatedcv", 
                             number = 10, 
                             repeats = 3,
                             savePredictions = TRUE, 
                             classProbs = TRUE)

# Ensemble the predictions of `models` to form a new combined prediction based on glm
stack.glm <- caretStack(models, 
                        method = "glm", 
                        metric = "Accuracy", 
                        trControl = stackControl)
print(stack.glm)

# Predict on testData
stack_predicteds <- predict(stack.glm, newdata=testData4)
head(stack_predicteds)
print(stack.glm)