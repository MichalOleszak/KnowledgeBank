devtools::install_github("tarashnot/SlopeOne")
devtools::install_github("tarashnot/SVDApproximation")
library(recommenderlab)
library(SlopeOne)
library(recosystem)
library(SVDApproximation)
library(dplyr)
library(data.table)

# *** overview *** ------------------------------------------------------------
# Veneficus recommender systems overveiw:
link <- paste0("https://veneficus-intranet.sharepoint.com/:w:/r/kennis/_layouts/",
               "15/Doc.aspx?sourcedoc=%7B3BB586E8-76A2-4DA4-B3F5-E760DE4C10F5%7D&file",
               "=Recommender%20systems%20overview.docx&action=default&mobileredirect=true")

# Comparison of different R packages:
# https://rpubs.com/tarashnot/recommender_comparison


# *** 1. recommenderlab *** ------------------------------------------------------
# Vignette:
# https://cran.r-project.org/web/packages/recommenderlab/vignettes/recommenderlab.pdf
#
# Collaborative filtering:
# Find similarity in users or items, using ratings of all/similar users/items.
# Only uses data on preferences/likes/ratings/clicks; no user or item characteristics.

# Handling data ---------------------------------------------------------------
# Example data matrix
m <- matrix(sample(c(1, NA), 50, replace = TRUE, prob = c(0.4, 0.6)), 
            ncol = 10, dimnames = list(user = paste("u", 1:5, sep = ""), 
                                       item = paste("i", 1:10, sep = "")))
# Transorm it to rating matrix
r <- as(m, "realRatingMatrix")
getRatingMatrix(r)
# Can also be converted to a data frame
head(as(r, "data.frame"))

# Binarize the matrix
r_b <- binarize(r, minRating = 1)
# Plot the matrix
image(r_b, main = "Clicks")

# Normalize by centering to remove rating bias
# (subtract the row mean from all ratings in the row)
# only applicable to real rating matrices, not binary ones!
r_m <- normalize(r)


# Creating a recommender ------------------------------------------------------
# Define the model:
#  * Similarity measurements: "Jaccard", "cosine", "pearson"
#  * Some methods: User- ("UBCF") or Item- ("IBCF") Based Collaborative Filtering

# Check available methods
recommenderRegistry$get_entry_names()
# Available methods for binary data
recommenderRegistry$get_entries(dataType = "binaryRatingMatrix")

# Define the model:
model <- Recommender(data = r_b, 
                     method = "UBCF", 
                     parameter = list(method = "cosine"))
getModel(model)

# Predict recommendations (in-sample, n items for each user)
recommendations_topN <- predict(model, r_b, n = 2)
# Predict raings or the whole matrix
recommendations_ratings <- predict(model, r_b, type = "ratings")
recommendations_ratMat <- predict(model, r_b, type = "ratingMatrix")
# See as list
getList(recommendations_topN)
# See as data frame
as(recommendations_topN, "matrix") %>% as.data.frame()


# Model evaluation ------------------------------------------------------------
# Evaluation starts with creating an evaluation scheme that determines what 
# and how data is used for training and testing. 
#
# How to split user into train and test sets:
#  - random split with set train/test sizes
#  - bootstrap sampling, good for small datasets
#  - k-fold CV
#
# For each user in the test data, some items are randomly withheld.
# For the "Given x" protocols, for each user, x randomly chosen items are given to 
# the recommender algorithm and the remaining items are withheld for evaluation.
#
# Example 1:
# Create an evaluation scheme which splits the data into a training set (80%)
# and a test set (20%). For the test set 2 items will be given to the
# recommender algorithm and the other items will be held out for computing the error.
eval <- evaluationScheme(r_b, method = "split", train = 0.8, given = 2, goodRating = 1)
# Example 2:
# A 4-fold cross validation scheme with all but three randomly selected items withhold 
# for the test users for evaluation
eval2 <- evaluationScheme(r_b, method = "cross", k = 4, given = 3, goodRating = 1)

# Create 2 recommenders (user- and item-based collaborative filtering) on training data
r1 <- Recommender(getData(eval, "train"), "UBCF")
r2 <- Recommender(getData(eval, "train"), "IBCF")

# Predicted ratings for the known part of the test data (2 items for each user)
p1 <- predict(r1, getData(eval, "known"), type = "topNList")
p2 <- predict(r2, getData(eval, "known"), type = "topNList")

# Evaluation of predicted ratings:
# Calculate the error between the prediction and the unknown part of the test data
error <- rbind(
  UBCF = calcPredictionAccuracy(p1, getData(eval, "unknown"), given = 2),
  IBCF = calcPredictionAccuracy(p2, getData(eval, "unknown"), given = 2)
) %>% print()

# Evaluation of a top-N recommender algorithm:
# Use the created evaluation scheme to evaluate the recommender
results <- evaluate(eval, method = "UBCF", type = "topNList", n = c(1, 3, 5, 10, 15, 20))
getConfusionMatrix(results)
avg(results)
# Evaluation results can be plotted using plot(). The default plot is the ROC curve which
# plots the true positive rate (TPR) against the false positive rate (FPR).
plot(results, annotate = TRUE)
# By using "prec/rec" as the second argument, a precision-recall plot is produced.
plot(results, "prec/rec", annotate = TRUE)


# Comparing algorithms --------------------------------------------------------
scheme <- evaluationScheme(r_b, method = "split", train = .8, k = 1, given = -1, goodRating = 5)
algorithms <- list(
  "random items"  =  list(name = "RANDOM", param = NULL),
  "popular items"  =  list(name = "POPULAR", param = NULL),
  "user-based CF"  =  list(name = "UBCF", param = list(nn = 50)),
  "item-based CF"  =  list(name = "IBCF", param = list(k = 50))
  # SVD does not work with binary data
  #"SVD approximation"  =  list(name = "SVD", param = list(k  =  50))
  )

# Compare topN
res <- evaluate(scheme, algorithms, type = "topNList", n = c(1, 3, 5, 10, 15, 20))
plot(results, annotate = c(1, 3), legend = "topleft")
plot(results, "prec/rec", annotate = 3, legend = "topleft")


# Hybrid recommender ----------------------------------------------------------
# Creates and combines recommendations using several recommender algorithms.
data("MovieLense")
MovieLense100 <- MovieLense[rowCounts(MovieLense) > 100,]
train <- MovieLense100[1:100]
test <- MovieLense100[101:103]

recom <- HybridRecommender(
  Recommender(train, method = "POPULAR"),
  Recommender(train, method = "RANDOM"),
  Recommender(train, method = "RERECOMMEND"),
  weights = c(.6, .1, .3)
)

recom
getModel(recom)
as(predict(recom, test), "list")



# *** 2. SlopeOne *** ------------------------------------------------------------
# Collaborative filtering
#
# Slope One method learns a set of simple predictors (one for each pair of two items) 
# with just constant variable. Therefore, this variable represents average difference 
# between ratings of two items. Using this method, fast computation and reasonable 
# accuracy could be easily achieved.
# Package SlopeOne works with data.table objects.

# Load exaple data
data(ratings)
head(ratings)

# Change names and types of variables of ratings dataset to make them suitable for package
names(ratings) <- c("user_id", "item_id", "rating")
ratings <- data.table(ratings)
ratings[, user_id := as.character(user_id)]
ratings[, item_id := as.character(item_id)]
setkey(ratings, user_id, item_id)

# Split data into train and test sets
in_train <- rep(TRUE, nrow(ratings))
in_train[sample(1:nrow(ratings), size = round(0.2 * length(unique(ratings$user_id)), 0) * 5)] <- FALSE
ratings_train <- ratings[(in_train)]
ratings_test <- ratings[(!in_train)]

# Normalize ratings
ratings_train_norm <- normalize_ratings(ratings_train)

# Build the model (slow)
model <- build_slopeone(ratings_train_norm$ratings)

# Making predictions for test set:
predictions <- predict_slopeone(model, 
                                ratings_test[ , c(1, 2), with = FALSE], 
                                ratings_train_norm$ratings)
unnormalized_predictions <- unnormalize_ratings(normalized = ratings_train_norm, 
                                                ratings = predictions)

# Check accuracy
rmse_slopeone <- sqrt(mean((unnormalized_predictions$predicted_rating - ratings_test$rating) ^ 2))



# *** 3. recosystem *** ----------------------------------------------------------
# Collaborative filtering
# Vignette: https://cran.r-project.org/web/packages/recosystem/vignettes/introduction.html
# 
# Matrix Factorization is a popular technique to solve recommender system problem. The main idea is 
# to approximate the matrix R(m x n) by the product of two matrixes of lower dimension: P (k x m) and Q (k x n).
# Matrix P represents latent factors of users. So, each k-elements column of matrix P represents each user. 
# Each k-elements column of matrix Q represents each item. To find rating for item i by user u we simply need 
# to compute two vectors: P[,u]' x Q[,i].


# Package usage ---------------------------------------------------------------
# 1. Create a model object (a Reference Class object in R) by calling Reco().
# 2. (Optionally) call the $tune() method to select best tuning parameters along a set of candidate values.
# 3. Train the model by calling the $train() method. A number of parameters can be set inside the function, 
#    possibly coming from the result of $tune().
# 4. (Optionally) output the model, i.e. write the factorized P and Q matrices info files.
# 5. Use the $predict() method to compute predictions and write results into a file.


# Example ---------------------------------------------------------------------
# Load simulated data
set.seed(123)
trainset = system.file("dat", "smalltrain.txt", package = "recosystem")
testset = system.file("dat", "smalltest.txt", package = "recosystem")

# Build recommender object and tune the parameters
# number of latent factors (dim), 
# gradient descend step rate (lrate), 
# penalty parameter to avoid overfitting (cost)
r = Reco()
opts = r$tune(trainset, opts = list(dim = c(10, 20, 30), 
                                    lrate = c(0.05, 0.1, 0.2),
                                    nthread = 1, 
                                    niter = 10))
# Train the model
r$train(trainset, opts = c(opts$min, nthread = 1, niter = 10))

# Get predictions
outfile = tempfile()
r$predict(testset, outfile)

# Compare the first few true values of testing data with predicted ones
# True values
print(read.table(testset, header = FALSE, sep = " ", nrows = 10)$V3)
# Predicted values
print(scan(outfile, n = 10))

# Calculate RMSE
scores_real <- read.table("testset.txt", header = FALSE, sep = " ")$V3
scores_pred <- scan(outfile)
(rmse_mf <- sqrt(mean((scores_real-scores_pred) ^ 2)))



# *** 4. SVDApproximation *** ----------------------------------------------------
# In this approach ranking matrix is decomposed based on Singular Value Decomposition 
# and then reconstructed keeping only first r entities. This gives an ability to predict 
# missing values. SVD Approximation method is similar to Matrix Factorization used by 
# recosystem, but here latent factors of users and items are retrieved in another way.


# Algorithm -------------------------------------------------------------------
# 1. Replace all missing values with items' averages;
# 2. Normalize matrix by subtracting users' averages (calculated based on initial rating matrix, not filled-in);
# 3. Perform Singular Value Decomposition of R;
# 4. Keeping only first r rows of matrix U, r rows and r columns of matrix S and r columns of matrix V, 
#    reconstruct matrix R: U[1:r,] x S[1:r,1:r] x V[,1:r]'


# Example ---------------------------------------------------------------------
# Prep data
set.seed(1)
mtx <- split_ratings(ratings_table = ratings, 
                     proportion = c(0.7, 0.15, 0.15))

# Build model
model <- svd_build(mtx)

# Tune r
# r denotes number of latent factors of decomposition and best value of this parameter 
# could be found using cross-validation or separate ratings for validation
model_tunes <- svd_tune(model, r = 2:50)
model_tunes$train_vs_valid

# Test on the test set
rmse_svd <- svd_rmse(model, r = model_tunes$r_best, rmse_type = c("test"))
