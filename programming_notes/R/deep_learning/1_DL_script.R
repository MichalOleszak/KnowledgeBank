# Settings --------------------------------------------------------------------
library(keras)
# use_condaenv("r-tensorflow", required = T)
# use_python("C:/Users/Asus/ANACON~1/envs/r-tensorflow", required = T)
# install_keras(method = "conda")

# MNIST example ---------------------------------------------------------------
mnist <- dataset_mnist()
train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y

network <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu", input_shape = c(28 * 28)) %>%
  layer_dense(units = 10, activation = "softmax")

network %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

train_images <- array_reshape(train_images, c(60000, 28 * 28))
train_images <- train_images / 255
test_images <- array_reshape(test_images, c(10000, 28 * 28))
test_images <- test_images / 255

train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

network %>% fit(train_images, train_labels, epochs = 5, batch_size = 128)

(metrics <- network %>% evaluate(test_images, test_labels))
network %>% predict_classes(test_images[1:10, ])


# Binary classification -------------------------------------------------------
imdb <- dataset_imdb(num_words = 10000)
c(c(train_data, train_labels),
  c(test_data, test_labels)) %<-% imdb

# Decoding reviews to English
word_index <- dataset_imdb_word_index()  # list mapping words to integer index
reverse_word_index <- names(word_index)
names(reverse_word_index) <- word_index  # reverse, mapping integers to words

decoded_review <- sapply(train_data[[2]], function(index) {
  word <- if (index >= 3) reverse_word_index[[as.numeric(index - 3)]]
  if (!is.null(word)) word else "?"      # offset by 3 as 0:2 are reserved indices
})

# Data prep: encode integer sequences into tesors (binary matrices)
# Goal: samples (reviews) in rows, one column for each word in the dictionary,
# full of zeros except for the words appearing in the correspoding review - these are 1
vectorize_sequences <- function(sequences, dimension = 10000) {
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences)) {
    results[i, sequences[[i]]] <- 1
  }
  return(results)
}

x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)
y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)

# Build the network
model <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

# Compile the model
model %>% compile(
  optimizer = "rmsprop",
  #optimizer = optimizer_rmsprop(lr = 0.001),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

# Validation set, to monitor out-of-sample accuracy during training
val_indices <- 1:10000
x_val <- x_train[val_indices, ]
partial_x_train <- x_train[-val_indices, ]
y_val <- y_train[val_indices]
partial_y_train <- y_train[-val_indices]

# Train the model
history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

# Highly overfitted model: after 4th epoch it's just memorizing training data
str(history)
plot(history)
history_df <- as.data.frame(history)

# Train another model with less epochs 
model <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)
model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 4,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)
(results <- model %>% evaluate(x_test, y_test))

# Use trained network to predict new data
model %>% predict(x_test[1:10, ])


# Multiclass classification ---------------------------------------------------
reuters <- dataset_reuters(num_words = 10000)
c(c(train_data, train_labels),
  c(test_data, test_labels)) %<-% reuters

# Decoding newswire to English
word_index <- dataset_reuters_word_index()  # list mapping words to integer index
reverse_word_index <- names(word_index)
names(reverse_word_index) <- word_index  # reverse, mapping integers to words

decoded_newswire <- sapply(train_data[[1]], function(index) {
  word <- if (index >= 3) reverse_word_index[[as.numeric(index - 3)]]
  if (!is.null(word)) word else "?"      # offset by 3 as 0:2 are reserved indices
})

# Data prep - vectorize the data
x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)

# Labels prep
# APPROACH 1: retain their integer values (just use them as they are)
# APPROACH 2: One-hot encoding (categorical encoding)
# Used for categorical data as an alternative to just casting the lables as integer vector
# Here: each label as a 0-vector with a 1 in the place of the label index
to_one_hot <- function(labels, dimension = 46) {
  results <- matrix(0, nrow = length(labels), ncol = dimension)
  for (i in 1:length(labels)) {
    results[i, labels[[i]] + 1] <- 1
  }
  return(results)
}
one_hot_train_labels <- to_one_hot(train_labels)
one_hot_test_labels <- to_one_hot(test_labels)
# This is equivalent to:
one_hot_train_labels <- to_categorical(train_labels)
one_hot_test_labels <- to_categorical(test_labels)

# Build the model
model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 46, activation = "softmax")

model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = "accuracy"
)

# Construct validation set
val_indices <- 1:1000
x_val <- x_train[val_indices, ]
partial_x_train <- x_train[-val_indices, ]
y_val <- one_hot_train_labels[val_indices, ]
partial_y_train <- one_hot_train_labels[-val_indices, ]

# Train the model
history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)
plot(history)
results <- model %>% evaluate(x_test, one_hot_test_labels)

# Overfitting after 9 epochs, so rerun the model with 9 epochs only
model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 46, activation = "softmax")
model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = "accuracy"
)
history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 9,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)
results <- model %>% evaluate(x_test, one_hot_test_labels)

# In balanced binary classification, a purely random classifier has accuracy of 50%.
# For 46 categories, it is around 18%, so the 79% by the network is quite ok.
test_labels_copy <- sample(test_labels)
mean(test_labels_copy == test_labels)

# Compute predictions
predictions <- model %>% predict(x_test)
dim(predictions)            # each entry of length 46
sum(predictions[1,])        # coefs sum up to 1
which.max(predictions[1,])  # the class with highest probability is the prediction

# Using labels as they are, without one_to_hot encoding
# In this case, sparse_categorical_crossentropy loss should be used
# Results are similar
y_val <- train_labels[val_indices]
partial_y_train <- train_labels[-val_indices]
model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 46, activation = "softmax")
model %>% compile(
  optimizer = "rmsprop",
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"
)
history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 9,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)
results <- model %>% evaluate(x_test, test_labels)


# Regression ------------------------------------------------------------------
boston <- dataset_boston_housing()
c(c(train_data, train_targets),
  c(test_data, test_targets)) %<-% boston

# Data prep: scale, because variables have different ranges
# Only use qunatities computed on training data!
train_mean <- apply(train_data, 2, mean)
train_std <- apply(train_data, 2, sd)
train_data <- scale(train_data, center = train_mean, scale = train_std)
test_data <- scale(test_data, center = train_mean, scale = train_std)

# Build model
build_model <- function() {
  model <- keras_model_sequential() %>% 
    layer_dense(units = 64, activation = "relu", input_shape = dim(train_data)[2]) %>% 
    layer_dense(units = 64, activation = "relu") %>% 
    layer_dense(units = 1)
  model %>% compile(
    optimizer = "rmsprop",
    loss = "mse",
    metrics = c("mae")
  )
}

# Validation via K-fold cross validation (as to few data points for separate validation set)
k <- 4
indices <- sample(1:nrow(train_data))
folds <- cut(indices, breaks = k, labels = FALSE)

num_epochs <- 500
all_mae_histories <- NULL
for (i in 1:k) {
  cat("processing fold #", i, "\n")
  # Prepare validation data from partiion #k
  val_indices <- which(folds == i, arr.ind = TRUE)
  val_data <- train_data[val_indices, ]
  val_targets <- train_targets[val_indices]
  # Prepare training data from all other partitions
  partial_train_data <- train_data[-val_indices, ]
  partial_train_targets <- train_targets[-val_indices]
  # Build a compiled keras model
  model <- build_model()
  # Train the model (in silent mode)
  history <- model %>% fit(partial_train_data, 
                           partial_train_targets,
                           validation_data = list(val_data, val_targets),
                           epochs = num_epochs,
                           batch_size = 1,
                           verbose = 0)
  # Evaluate the model on validation data
  mae_history <- history$metrics$val_mean_absolute_error
  all_mae_histories <- rbind(all_mae_histories, mae_history)
}

# Average per-epoch MAE for all folds
average_mae_history <- data.frame(
  epoch = seq(1:ncol(all_mae_histories)),
  validation_mae = apply(all_mae_histories, 2, mean)
)
# Overfitting starts after 125th epoch
ggplot(average_mae_history, aes(epoch, validation_mae)) +
  # geom_line()
  geom_smooth()

# After tweaking parameters with cross-validatoin, train the final model on all data
model <- build_model()
model %>% fit(train_data, 
              train_targets,
              epochs = 80,
              batch_size = 16,
              verbose = 0)
(result <- model %>% evaluate(test_data, test_targets))


# Preventing overfitting ------------------------------------------------------
# 1. Get more training data - naturally better generalization
# 2. Reduce network's capacity - less layers / nodes, but not too little
# 3. Add regularization:
#     - L1: the cost proportional to absolute value of the weights
#     - L2 (weight decay): the cost proportional to squared weight value
# 4. Add dropout: 
#    Dropout, applied to a layer, consists of randomly dropping out (setting to zero)
#    a number of output features of the layer during training. Dropout rate usually
#    between 0.2 and 0.5. At testing, nothing is zeroed: instead, the layer's output
#    values are scaled down by the factor equal to the dropout rate, to balance 
#    for the fact that more units are active than at the training time.
#    In practice, both dropping and scaling done at training, test remains unchanged.

# Adding regularization:
model <- keras_model_sequential() %>% 
  layer_dense(units = 16, kernel_regularizer = regularizer_l2(0.001),
              activation = "relu", input_shape = c(10000)) %>% 
  layer_dense(units = 16, kernel_regularizer = regularizer_l2(0.001),
              activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")
# Penalty only added at training time, so higher training loss than test loss
# available: regularizer_l2(), regularizer_l1(), regularizer_l1_l2(l1 = x, l2 = y)

# Adding dropout:
model <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = "sigmoid") 


# Universal ML Workflow -------------------------------------------------------
# 1. Define the problem & assemble data
#     Binary/Muticlass (Multilabel) Classification? Scalar/Vector regression?
# 2. Choose success measure
#     Precision/Recall? Accuracy? AUC? The metric will guide the choice of the 
#     loss function - what the model will optimize. It should directly allign 
#     with the higher-level goals, such as the success of the business.
# 3. Decide on an evaluation protocol
#     - maintain a hold-out validation set: way to go when there are plenty of data
#     - K-fold cross-validation: when too little data for a hold-out validation set
#     - iterated K-fold cross-validation: highly accurate with little data available
# 4. Prep the data
#     - formatted as tensors
#     - values scaled to small values, e.g. [-1,1] or [0,1] range
#     - if ranges are different across features, normalize the data
#     - some feature engineering?
# 5. Develop a model that does better than a naive baseline
#     Problem type / last-layer activation / loss function:
#     Binary classification / sigmoid / binary_crossentropy
#     Multiclass, single-label classification / softmax / categorical_crossentropy
#     Multiclass, multilabel classification / sigmoid / binary_crossentropy
#     Regression to arbitrary values / none / mse
#     Regression to [0,1] values / sigmoid / mse or binary_crossentropy
# 6. Develop a model that overfits
#     - add layers
#     - make them bigger
#     - tran for more epochs
# 7. Regularize the model and tune hyperparameters
#     - add dropout
#     - add/remove layers
#     - add L1 and/or L2 regularization
#     - try different hyperparams (# units per layer, optimizer's learning rate)
#     - iterate on feature engineering: add new features, remove uninformative ones












