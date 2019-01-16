# *** WORKING WITH TEXT DATA  *** ----

# One-hot encoding of words ---------------------------------------------------
# It consists in associating a unique integer index to every word, then turning 
# this integer index i into a binary vector of size N, the size of the vocabulary, 
# that would be all-zeros except for the i-th entry, which would be 1.

# Word level one-hot encoding (toy example):
# This is our initial data; one entry per "sample"
# (in this toy example, a "sample" is just a sentence, but
# it could be an entire document).
samples <- c("The cat sat on the mat.", "The dog ate my homework.")

# First, build an index of all tokens in the data.
token_index <- list()
for (sample in samples)
  # Tokenizes the samples via the strsplit function. In real life, you'd also
  # strip punctuation and special characters from the samples.
  for (word in strsplit(sample, " ")[[1]])
    if (!word %in% names(token_index))
      # Assigns a unique index to each unique word. Note that you don't
      # attribute index 1 to anything.
      token_index[[word]] <- length(token_index) + 2 
# Vectorizes the samples. You'll only consider the first max_length 
# words in each sample.
max_length <- 10
# This is where you store the results.
results <- array(0, dim = c(length(samples), 
                            max_length, 
                            max(as.integer(token_index))))
for (i in 1:length(samples)) {
  sample <- samples[[i]]
  words <- head(strsplit(sample, " ")[[1]], n = max_length)
  for (j in 1:length(words)) {
    index <- token_index[[words[[j]]]]
    results[[i, j, index]] <- 1
  }
}


# One-hot encoding of characters ----------------------------------------------
# Character level one-hot encoding (toy example):
samples <- c("The cat sat on the mat.", "The dog ate my homework.")
ascii_tokens <- c("", sapply(as.raw(c(32:126)), rawToChar))
token_index <- c(1:(length(ascii_tokens)))
names(token_index) <- ascii_tokens
max_length <- 50
results <- array(0, dim = c(length(samples), max_length, length(token_index)))
for (i in 1:length(samples)) {
  sample <- samples[[i]]
  characters <- strsplit(sample, "")[[1]]
  for (j in 1:length(characters)) {
    character <- characters[[j]]
    results[i, j, token_index[[character]]] <- 1
  }
}

# Using Keras for word-level one-hot encoding:
library(keras)
samples <- c("The cat sat on the mat.", "The dog ate my homework.")
# Creates a tokenizer, configured to only take into account the 1,000 
# most common words, then builds the word index.
tokenizer <- text_tokenizer(num_words = 1000) %>%
  fit_text_tokenizer(samples)
# Turns strings into lists of integer indices
sequences <- texts_to_sequences(tokenizer, samples)
# You could also directly get the one-hot binary representations. Vectorization 
# modes other than one-hot encoding are supported by this tokenizer.
one_hot_results <- texts_to_matrix(tokenizer, samples, mode = "binary")
# How you can recover the word index that was computed
word_index <- tokenizer$word_index
cat("Found", length(word_index), "unique tokens./n")



# One-hot hashing trick -------------------------------------------------------
# A variant of one-hot encoding is the so-called “one-hot hashing trick”, which can be 
# used when the number of unique tokens in your vocabulary is too large to handle 
# explicitly. Instead of explicitly assigning an index to each word and keeping 
# a reference of these indices in a dictionary, one may hash words into vectors of 
# fixed size. This is typically done with a very lightweight hashing function. 
# The main advantage of this method is that it does away with maintaining an explicit
# word index, which saves memory and allows online encoding of the data (starting to 
# generate token vectors right away, before having seen all of the available data). 
# The one drawback of this method is that it is susceptible to “hash collisions”: 
# two different words may end up with the same hash, and subsequently any machine 
# learning model looking at these hashes won’t be able to tell the difference between 
# these words. The likelihood of hash collisions decreases when the dimensionality of 
# the hashing space is much larger than the total number of unique tokens being hashed.

# Word-level one-hot encoding with hashing trick (toy example):
library(hashFunction)
samples <- c("The cat sat on the mat.", "The dog ate my homework.")
# We will store our words as vectors of size 1000.
# Note that if you have close to 1000 words (or more)
# you will start seeing many hash collisions, which
# will decrease the accuracy of this encoding method.
dimensionality <- 1000
max_length <- 10
results <- array(0, dim = c(length(samples), max_length, dimensionality))
for (i in 1:length(samples)) {
  sample <- samples[[i]]
  words <- head(strsplit(sample, " ")[[1]], n = max_length)
  for (j in 1:length(words)) {
    # Hash the word into a "random" integer index
    # that is between 0 and 1,000
    index <- abs(spooky.32(words[[i]])) %% dimensionality
    results[[i, j, index]] <- 1
  }
}


# Using word embeddings -------------------------------------------------------
# Another popular and powerful way to associate a vector with a word is the use of dense
# “word vectors”, also called “word embeddings”. While the vectors obtained through 
# one-hot encoding are binary, sparse (mostly made of zeros) and very high-dimensional 
# (same dimensionality as the number of words in the vocabulary), “word embeddings” are 
# low-dimensional floating point vectors (i.e. “dense” vectors, as opposed to sparse 
# vectors). Unlike word vectors obtained via one-hot encoding, word embeddings are 
# learned from data. It is common to see word embeddings that are 256-dimensional, 
# 512-dimensional, or 1024-dimensional when dealing with very large vocabularies. 
# On the other hand, one-hot encoding words generally leads to vectors that are 
# 20,000-dimensional or higher (capturing a vocabulary of 20,000 token in this case). 
# So, word embeddings pack more information into far fewer dimensions.
#
# There are two ways to obtain word embeddings:
#  1. Learn word embeddings jointly with the main task you care about (e.g. document 
#     classification or sentiment prediction). In this setup, you would start with random 
#     word vectors, then learn your word vectors in the same way that you learn the weights
#     of a neural network.
#  2. Load into your model word embeddings that were pre-computed using a different machine
#     learning task than the one you are trying to solve. These are called “pre-trained 
#     word embeddings”.

# 1. Learning word embeddings ----
# The embedding layer takes at least two arguments:
#  - the number of possible tokens, here 1000 (1 + maximum word index),
#  - and the dimensionality of the embeddings, here 64.
embedding_layer <- layer_embedding(input_dim = 1000, output_dim = 64) 
# A layer_embedding() is best understood as a dictionary that maps integer indices
# (which stand for specific words) to dense vectors. It takes integers as input, it 
# looks up these integers in an internal dictionary, and it returns the associated 
# vectors. It’s effectively a dictionary lookup.
#
# An embedding layer takes as input a 2D tensor of integers, of shape (samples, sequence_length),
# where each entry is a sequence of integers. This layer returns a 3D floating-point tensor, 
# of shape (samples, sequence_length, embedding_dimensionality). Such a 3D tensor can then 
# be processed by an RNN layer or a 1D convolution layer.

# IMDB example:
# Number of words to consider as features
max_features <- 10000
# Cut texts after this number of words 
# (among top max_features most common words)
maxlen <- 20
# Load the data as lists of integers.
imdb <- dataset_imdb(num_words = max_features)
c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb
# This turns our lists of integers
# into a 2D integer tensor of shape `(samples, maxlen)`
x_train <- pad_sequences(x_train, maxlen = maxlen)
x_test <- pad_sequences(x_test, maxlen = maxlen)

model <- keras_model_sequential() %>% 
  # We specify the maximum input length to our Embedding layer
  # so we can later flatten the embedded inputs
  layer_embedding(input_dim = 10000, output_dim = 8, 
                  input_length = maxlen) %>% 
  # We flatten the 3D tensor of embeddings 
  # into a 2D tensor of shape `(samples, maxlen * 8)`
  layer_flatten() %>% 
  # We add the classifier on top
  layer_dense(units = 1, activation = "sigmoid") 

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)
history <- model %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2
)


# 2. Using pre-trained word embeddings ----
# We will be using a model similar to the one we just went over – embedding sentences in 
# sequences of vectors, flattening them and training a dense layer on top. But we will do 
# it using pre-trained word embeddings, and instead of using the pre-tokenized IMDB data 
# packaged in Keras, we will start from scratch, by downloading the original text data.

# Download the IMDB data as raw text
imdb_dir <- "C:/Users/Asus/Dropbox/R/Deep Learning with R/data/aclImdb"
train_dir <- file.path(imdb_dir, "train")
labels <- c()
texts <- c()
for (label_type in c("neg", "pos")) {
  label <- switch(label_type, neg = 0, pos = 1)
  dir_name <- file.path(train_dir, label_type)
  for (fname in list.files(dir_name, pattern = glob2rx("*.txt"), 
                           full.names = TRUE)) {
    texts <- c(texts, readChar(fname, file.info(fname)$size))
    labels <- c(labels, label)
  }
}

# Tokenize the data
maxlen <- 100                 # We will cut reviews after 100 words
training_samples <- 200       # We will be training on 200 samples
validation_samples <- 10000   # We will be validating on 10000 samples
max_words <- 10000            # We will only consider the top 10,000 words in the dataset
tokenizer <- text_tokenizer(num_words = max_words) %>% 
  fit_text_tokenizer(texts)
sequences <- texts_to_sequences(tokenizer, texts)
word_index = tokenizer$word_index
cat("Found", length(word_index), "unique tokens./n")

data <- pad_sequences(sequences, maxlen = maxlen)
labels <- as.array(labels)
cat("Shape of data tensor:", dim(data), "/n")
cat('Shape of label tensor:', dim(labels), "/n")

# Split the data into a training set and a validation set
# But first, shuffle the data, since we started from data
# where sample are ordered (all negative first, then all positive).
indices <- sample(1:nrow(data))
training_indices <- indices[1:training_samples]
validation_indices <- indices[(training_samples + 1): 
                                (training_samples + validation_samples)]
x_train <- data[training_indices,]
y_train <- labels[training_indices]
x_val <- data[validation_indices,]
y_val <- labels[validation_indices]

# Download the GloVe word embeddings
# Head to https://nlp.stanford.edu/projects/glove/ (where you can learn more about the 
# GloVe algorithm), and download the pre-computed embeddings from 2014 English Wikipedia. 
# It’s a 822MB zip file named glove.6B.zip, containing 100-dimensional embedding vectors 
# for 400,000 words (or non-word tokens). Un-zip it.

# Pre-process the embeddings
# parse the un-zipped file (it’s a txt file) to build an index
# mapping words (as strings) to their vector representation (as number vectors).
glove_dir = '~/data/glove.6B'
lines <- readLines(file.path(glove_dir, "glove.6B.100d.txt"))
embeddings_index <- new.env(hash = TRUE, parent = emptyenv())
for (i in 1:length(lines)) {
  line <- lines[[i]]
  values <- strsplit(line, " ")[[1]]
  word <- values[[1]]
  embeddings_index[[word]] <- as.double(values[-1])
}

# Build an embedding matrix that you can load into an embedding layer. It must be a matrix 
# of shape (max_words, embedding_dim), where each entry i contains the embedding_dim-dimensional
# vector for the word of index i in the reference word index (built during tokenization). 
# Note that index 1 isn’t supposed to stand for any word or token – it’s a placeholder.
embedding_dim <- 100
embedding_matrix <- array(0, c(max_words, embedding_dim))
for (word in names(word_index)) {
  index <- word_index[[word]]
  if (index < max_words) {
    embedding_vector <- embeddings_index[[word]]
    if (!is.null(embedding_vector))
      # Words not found in the embedding index will be all zeros.
      embedding_matrix[index+1,] <- embedding_vector
  }
}

# Define a model
model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_words, output_dim = embedding_dim, 
                  input_length = maxlen) %>% 
  layer_flatten() %>% 
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")
summary(model)

# Load the GloVe embeddings in the model
get_layer(model, index = 1) %>% 
  set_weights(list(embedding_matrix)) %>% 
  freeze_weights()

# Train and evaluate
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)
history <- model %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 32,
  validation_data = list(x_val, y_val)
)
save_model_weights_hdf5(model, "pre_trained_glove_model.h5")

# The model quickly starts overfitting, unsurprisingly given the small 
# number of training samples. We can also try to train the same model without 
# loading the pre-trained word embeddings and without freezing the embedding layer. 
# In that case, we would be learning a task-specific embedding of our input tokens, 
# which is generally more powerful than pre-trained word embeddings when lots of data is 
# available.
model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_words, output_dim = embedding_dim, 
                  input_length = maxlen) %>% 
  layer_flatten() %>% 
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)
history <- model %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 32,
  validation_data = list(x_val, y_val)
)

# Evaluate the model on the test data. First, we will need to tokenize the test data:
test_dir <- file.path(imdb_dir, "test")
labels <- c()
texts <- c()
for (label_type in c("neg", "pos")) {
  label <- switch(label_type, neg = 0, pos = 1)
  dir_name <- file.path(test_dir, label_type)
  for (fname in list.files(dir_name, pattern = glob2rx("*.txt"), 
                           full.names = TRUE)) {
    texts <- c(texts, readChar(fname, file.info(fname)$size))
    labels <- c(labels, label)
  }
}
sequences <- texts_to_sequences(tokenizer, texts)
x_test <- pad_sequences(sequences, maxlen = maxlen)
y_test <- as.array(labels)

model %>% 
  load_model_weights_hdf5("pre_trained_glove_model.h5") %>% 
  evaluate(x_test, y_test, verbose = 0)


# *** RECURRENT NEURAL NETWORKS *** ----

# RNN pseudocode --------------------------------------------------------------
# RNN iterates through sequence elements and maintains the state info about what
# it has seen so far.
state_t <- 0
for (input_t in input_sequence) {
  output_t <- activation(dot(W, input_t) + dot(U, state_t) + b)
  state_t <- output_t
}


# Simple RNN - manual implementation -------------------------------------------------
timesteps <- 100 # number of timesteps in input sequence
input_features <- 32 # dimensionality of the input feature space
output_features <- 64 # dimensionality of the output feature space

random_array <- function(dim) {
  array(runif(prod(dim)), dim = dim)
}
inputs <- random_array(dim = c(timesteps, input_features)) # random example data
state_t <- rep_len(0, length = c(output_features)) # initial state - an all-zero vector

W <- random_array(dim = c(output_features, input_features)) # random weight matrices
U <- random_array(dim = c(output_features, output_features))
b <- random_array(dim = c(output_features, 1))

output_sequence <- array(0, dim = c(timesteps, output_features))
for (i in 1:nrow(inputs)) {
  input_t <- inputs[i, ] # input_t is a vector of shape (input_features)
  # combine input with the current state (previous output) to obtain current output
  output_t <- tanh(as.numeric((W %*% input_t) + (U %*% state_t) + b))
  output_sequence[i, ] <- as.numeric(output_t) # update result matrix
  state_t <- output_t # update network's state for the next timestep
}

# RNN with keras --------------------------------------------------------------
# The process you just naively implemented in R corresponds to an actual Keras layer:
# layer_simple_rnn(). There is one minor difference: layer_simple_rnn() processes batches 
# of sequences, like all other Keras layers, not a single sequence as in the R example. 
# This means it takes inputs of shape (batch_size, timesteps, input_features), rather than 
# (timesteps, input_features).
# 
# Like all recurrent layers in Keras, layer_simple_rnn() can be run in two different 
# modes: it can return either the full sequences of successive outputs for each timestep
# (a 3D tensor of shape (batch_size, timesteps, output_features)) or only the last output 
# for each input sequence (a 2D tensor of shape (batch_size, output_features)). 
# These two modes are controlled by the return_sequences constructor argument. 

model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = 10000, output_dim = 32) %>% 
  layer_simple_rnn(units = 32)
summary(model)

model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = 10000, output_dim = 32) %>% 
  layer_simple_rnn(units = 32, return_sequences = TRUE)
summary(model)

# It is sometimes useful to stack several recurrent layers one after the other in order to 
# increase the representational power of a network. In such a setup, you have to get all 
# intermediate layers to return full sequences:
model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = 10000, output_dim = 32) %>% 
  layer_simple_rnn(units = 32, return_sequences = TRUE) %>% 
  layer_simple_rnn(units = 32, return_sequences = TRUE) %>%
  layer_simple_rnn(units = 32, return_sequences = TRUE) %>%
  layer_simple_rnn(units = 32)  # This last layer only returns the last outputs.
summary(model)

# IMDB example
# Data prep
max_features <- 10000  # Number of words to consider as features
maxlen <- 500  # Cuts off texts after this many words (among the max_features most common words)
batch_size <- 32
cat("Loading data...\n")
imdb <- dataset_imdb(num_words = max_features)
c(c(input_train, y_train), c(input_test, y_test)) %<-% imdb 
cat(length(input_train), "train sequences\n")
cat(length(input_test), "test sequences")
cat("Pad sequences (samples x time)\n")
input_train <- pad_sequences(input_train, maxlen = maxlen)
input_test <- pad_sequences(input_test, maxlen = maxlen)
cat("input_train shape:", dim(input_train), "\n")
cat("input_test shape:", dim(input_test), "\n")

# Model
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, output_dim = 32) %>%
  layer_simple_rnn(units = 32) %>%
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)
history <- model %>% fit(
  input_train, y_train,
  epochs = 10,
  batch_size = 128,
  validation_split = 0.2
)
# layer_simple_rnn() isn’t good at processing long sequences, such as text;
# better to use LSTM or GRU layers.


# LSTM / GRU --------------------------------------------------------------------------------------
# simple RNNs suffer from vanishing gradient problem: too many layers make nets untrainable
# LSTM and GRU seek to alleviate it
# LSTM us a variant of layer_simple_rnn which adds a way to carry info across many timesteps

# LSTM psuedocode ---------------------------------------------------------------------------------
output_t <- activation(dot(state_t, U_o) + dot(input_t, W_o) + dot(C_t, V_o) + b_o)

i_t <- activation(dot(state_t, Ui) + dot(input_t, Wi) + bi)
f_t <- activation(dot(state_t, Uf) + dot(input_t, Wf) + bf)
k_t <- activation(dot(state_t, Uk) + dot(input_t, Wk) + bk)
# new carry state:
c_t+1 <- i_t * k_t + c_t * f_t


# LSTM example ------------------------------------------------------------------------------------
model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_features, output_dim = 32) %>% 
  layer_lstm(units = 32) %>% 
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  optimizer = "rmsprop", 
  loss = "binary_crossentropy", 
  metrics = c("acc")
)
history <- model %>% fit(
  input_train, y_train,
  epochs = 10,
  batch_size = 128,
  validation_split = 0.2
)


# *** ADVANCED USAGE OF RNNs *** ----

# Three advanced techniques:
# - Recurrent dropout, a specific, built-in way to use dropout to fight overfitting in recurrent layers.
# - Stacking recurrent layers, to increase the representational power of the network 
#   (at the cost of higher computational loads).
# - Bidirectional recurrent layers, which presents the same information to a recurrent network in 
#   different ways, increasing accuracy and mitigating forgetting issues.

# Download temperature data
#dir.create("C:/Users/Asus/Dropbox/R/Deep Learning with R/data/jena_climate", recursive = TRUE)
#download.file(
#  "https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip",
#  "C:/Users/Asus/Dropbox/R/Deep Learning with R/data/jena_climate/jena_climate_2009_2016.csv.zip"
#)
#unzip(
#  "C:/Users/Asus/Dropbox/R/Deep Learning with R/data/jena_climate/jena_climate_2009_2016.csv.zip",
#  exdir = "C:/Users/Asus/Dropbox/R/Deep Learning with R/data/jena_climate"
#)

library(tibble)
library(readr)
data_dir <- "C:/Users/Asus/Dropbox/R/Deep Learning with R/data/jena_climate"
fname <- file.path(data_dir, "jena_climate_2009_2016.csv")
data <- read_csv(fname)
glimpse(data)

library(ggplot2)
ggplot(data, aes(x = 1:nrow(data), y = `T (degC)`)) + geom_line()
ggplot(data[1:1440,], aes(x = 1:1440, y = `T (degC)`)) + geom_line()


# The exact formulation of our problem will be the following: given data going as far back as lookback 
# timesteps (a timestep is 10 minutes) and sampled every steps timesteps, can we predict the temperature 
# in delay timesteps?
#
# We will use the following parameter values:
# lookback = 1440, i.e. our observations will go back 10 days.
# steps = 6, i.e. our observations will be sampled at one data point per hour.
# delay = 144, i.e. our targets will be 24 hours in the future.

# To get started, we need to do two things:
# - Preprocess the data to a format a neural network can ingest. This is easy: the data is already 
#   numerical, so we don’t need to do any vectorization. However each timeseries in the data is on a 
#   different scale (e.g. temperature is typically between -20 and +30, but pressure, measured in mbar, 
#   is around 1000). So we will normalize each timeseries independently so that they all take small values 
#   on a similar scale.
# - Write a generator function that takes the current array of float data and yields batches of data from 
#   the recent past, along with a target temperature in the future. Because the samples in the dataset are
#   highly redundant (sample N and sample N + 1 will have most of their timesteps in common), it would be 
#   wasteful to explicitly allocate every sample. Instead, you’ll generate the samples on the fly using 
#   the original data.

data <- data.matrix(data[,-1])

train_data <- data[1:200000,]
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
data <- scale(data, center = mean, scale = std)

# Now here is the data generator you’ll use. It yields a list (samples, targets), where samples is one 
# batch of input data and targets is the corresponding array of target temperatures. It takes the 
# following arguments:
#  - data – The original array of floating-point data, which you normalized above.
#  - lookback – How many timesteps back the input data should go.
#  - delay – How many timesteps in the future the target should be.
#  - min_index and max_index – Indices in the data array that delimit which timesteps to draw from. 
#    This is useful for keeping a segment of the data for validation and another for testing.
#  - shuffle – Whether to shuffle the samples or draw them in chronological order.
#  - batch_size – The number of samples per batch.
#  - step – The period, in timesteps, at which you sample data. You’ll set it 6 in order to draw one 
#    data point every hour.

generator <- function(data, lookback, delay, min_index, max_index,
                      shuffle = FALSE, batch_size = 128, step = 6) {
  if (is.null(max_index))
    max_index <- nrow(data) - delay - 1
  i <- min_index + lookback
  function() {
    if (shuffle) {
      rows <- sample(c((min_index+lookback):max_index), size = batch_size)
    } else {
      if (i + batch_size >= max_index)
        i <<- min_index + lookback
      rows <- c(i:min(i+batch_size, max_index))
      i <<- i + length(rows)
    }
    
    samples <- array(0, dim = c(length(rows), 
                                lookback / step,
                                dim(data)[[-1]]))
    targets <- array(0, dim = c(length(rows)))
    
    for (j in 1:length(rows)) {
      indices <- seq(rows[[j]] - lookback, rows[[j]], 
                     length.out = dim(samples)[[2]])
      samples[j,,] <- data[indices,]
      targets[[j]] <- data[rows[[j]] + delay,2]
    }            
    
    list(samples, targets)
  }
}

lookback <- 1440
step <- 6
delay <- 144
batch_size <- 128
train_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 1,
  max_index = 200000,
  shuffle = TRUE,
  step = step, 
  batch_size = batch_size
)
val_gen = generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 200001,
  max_index = 300000,
  step = step,
  batch_size = batch_size
)
test_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 300001,
  max_index = NULL,
  step = step,
  batch_size = batch_size
)
# This is how many steps to draw from `val_gen`
# in order to see the whole validation set:
val_steps <- (300000 - 200001 - lookback) / batch_size
# This is how many steps to draw from `test_gen`
# in order to see the whole test set:
test_steps <- (nrow(data) - 300001 - lookback) / batch_size

# A common sense, non-machine learning baseline: 
# to always predict that the temperature 24 hours from now will be equal to the temperature right now
evaluate_naive_method <- function() {
  batch_maes <- c()
  for (step in 1:val_steps) {
    c(samples, targets) %<-% val_gen()
    preds <- samples[,dim(samples)[[2]],2]
    mae <- mean(abs(preds - targets))
    batch_maes <- c(batch_maes, mae)
  }
  print(mean(batch_maes))
}

mae <- evaluate_naive_method()
# Since our temperature data has been normalized to be centered on 0 and have a standard deviation 
# of one, this number is not immediately interpretable. It translates to an average absolute error 
# of (mae * temperature_std) ~ 2.5 degrees Celsius

# A basic machine learning approach: a small, densely connected network
model <- keras_model_sequential() %>% 
  layer_flatten(input_shape = c(lookback / step, dim(data)[-1])) %>% 
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)
history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 20,
  validation_data = val_gen,
  validation_steps = val_steps
)

# A first recurrent baseline
model <- keras_model_sequential() %>% 
  layer_gru(units = 32, input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_dense(units = 1)
model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)
history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 20,
  validation_data = val_gen,
  validation_steps = val_steps
)

# Recurrent dropout to fight overfitting ----------------------------------------------------------

# Regular dropout consists in randomly zeroing-out input units of a layer in order to break 
# happenstance correlations in the training data that the layer is exposed to. 
#
# How to correctly apply dropout in recurrent networks?
#
# The same dropout mask (the same pattern of dropped units) should be applied at every timestep, 
# instead of a dropout mask that would vary randomly from timestep to timestep. What’s more: in order 
# to regularize the representations formed by the recurrent gates of layers such as GRU and LSTM, a 
# temporally constant dropout mask should be applied to the inner recurrent activations of the layer
# (a “recurrent” dropout mask). Using the same dropout mask at every timestep allows the network to 
# properly propagate its learning error through time; a temporally random dropout mask would instead
# disrupt this error signal and be harmful to the learning process.

model <- keras_model_sequential() %>% 
  layer_gru(units = 32, dropout = 0.2, recurrent_dropout = 0.2,
            input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_dense(units = 1)
model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)
# Because networks being regularized with dropout always take longer to fully converge, 
# we train our network for twice as many epochs.
history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 40, #
  validation_data = val_gen,
  validation_steps = val_steps
)

# Stacking recurrent layers -----------------------------------------------------------------------

# o stack recurrent layers on top of each other in Keras, all intermediate layers should return 
# their full sequence of outputs (a 3D tensor) rather than their output at the last timestep. 
# This is done by specifying return_sequences = TRUE.

model <- keras_model_sequential() %>% 
  layer_gru(units = 32, 
            dropout = 0.1, 
            recurrent_dropout = 0.5,
            return_sequences = TRUE,
            input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_gru(units = 64, activation = "relu",
            dropout = 0.1,
            recurrent_dropout = 0.5) %>% 
  layer_dense(units = 1)
model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)
history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 40,
  validation_data = val_gen,
  validation_steps = val_steps
)

# Bidirectional RNNs ------------------------------------------------------------------------------

# RNNs are notably order-dependent, or time-dependent: they process the timesteps of their input 
# sequences in order, and shuffling or reversing the timesteps can completely change the representations 
# that the RNN will extract from the sequence. This is precisely the reason why they perform well on 
# problems where order is meaningful, such as our temperature forecasting problem. A bidirectional RNN 
# exploits the order-sensitivity of RNNs: it simply consists of two regular RNNs, such as the GRU or LSTM 
# layers that you are already familiar with, each processing input sequence in one direction 
# (chronologically and antichronologically), then merging their representations. By processing a sequence
# both way, a bidirectional RNN is able to catch patterns that may have been overlooked by a one-direction 
# RNN.

# Train model by feeding data in reverse order (newer first)
reverse_order_generator <- function( data, lookback, delay, min_index, max_index,
                                     shuffle = FALSE, batch_size = 128, step = 6) {
  if (is.null(max_index))
    max_index <- nrow(data) - delay - 1
  i <- min_index + lookback
  function() {
    if (shuffle) {
      rows <- sample(c((min_index+lookback):max_index), size = batch_size)
    } else {
      if (i + batch_size >= max_index)
        i <<- min_index + lookback
      rows <- c(i:min(i+batch_size, max_index))
      i <<- i + length(rows)
    }
    
    samples <- array(0, dim = c(length(rows), 
                                lookback / step,
                                dim(data)[[-1]]))
    targets <- array(0, dim = c(length(rows)))
    
    for (j in 1:length(rows)) {
      indices <- seq(rows[[j]] - lookback, rows[[j]], 
                     length.out = dim(samples)[[2]])
      samples[j,,] <- data[indices,]
      targets[[j]] <- data[rows[[j]] + delay,2]
    }            
    
    list(samples[,ncol(samples):1,], targets)
  }
}
train_gen_reverse <- reverse_order_generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 1,
  max_index = 200000,
  shuffle = TRUE,
  step = step, 
  batch_size = batch_size
)
val_gen_reverse = reverse_order_generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 200001,
  max_index = 300000,
  step = step,
  batch_size = batch_size
)

model <- keras_model_sequential() %>% 
  layer_gru(units = 32, input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_dense(units = 1)
model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)
history <- model %>% fit_generator(
  train_gen_reverse,
  steps_per_epoch = 500,
  epochs = 20,
  validation_data = val_gen_reverse,
  validation_steps = val_steps
)

# the reversed-order GRU strongly underperforms even the common-sense baseline, indicating that 
# the in our case chronological processing is very important to the success of our approach. 
# This makes perfect sense: the underlying GRU layer will typically be better at remembering the
# recent past than the distant past, and naturally the more recent weather data points are more
# predictive than older data points in our problem.
#
# This is generally not true for many other problems, including natural language: intuitively, 
# the importance of a word in understanding a sentence is not usually dependent on its position 
# in the sentence.

# Try the same trick on the LSTM IMDB example
max_features <- 10000  # Number of words to consider as features
maxlen <- 500          # Cut texts after this number of words 
# (among top max_features most common words)
# Load data
imdb <- dataset_imdb(num_words = max_features)
c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb
# Reverse sequences
x_train <- lapply(x_train, rev) 
x_test <- lapply(x_test, rev) 
# Pad sequences
x_train <- pad_sequences(x_train, maxlen = maxlen)
x_test <- pad_sequences(x_test, maxlen = maxlen)
model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_features, output_dim = 128) %>% 
  layer_lstm(units = 32) %>% 
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

history <- model %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 128,
  validation_split = 0.2
)

# We get near-identical performance as the chronological-order LSTM we tried in the previous section.

# A bidirectional RNN exploits this idea to improve upon the performance of chronological-order RNNs: 
# it looks at its inputs sequence both ways, obtaining potentially richer representations and capturing
# patterns that may have been missed by the chronological-order version alone.
k_clear_session()

model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_features, output_dim = 32) %>% 
  bidirectional(
    layer_lstm(units = 32)
  ) %>% 
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)
history <- model %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 128,
  validation_split = 0.2
)
# It performs slightly better than the regular LSTM.  It also seems to overfit faster, which is
# unsurprising since a bidirectional layer has twice more parameters than a chronological LSTM.


# *** SEQUENCE PROCESSING WITH CONVNETS *** ----

# 1D convolutions ---------------------------------------------------------------------------------
# 1D convnet are a fast alternative to RNNs. 1D convolution can recognize local patterns in a sequence
# and recognize them at a different position in the sequence.

# In Keras, you use a 1D convnet via the layer_conv_1d() function, which has an interface similar 
# to layer_conv_2d(). It takes as input 3D tensors with shape (samples, time, features) and returns 
# similarly shaped 3D tensors. The convolution window is a 1D window on the temporal axis: the second 
# axis in the input tensor.

# A simple two-layer 1D convnet and apply it to the IMDB sentiment-classification task
max_features <- 10000
max_len <- 500
cat("Loading data...\n")
imdb <- dataset_imdb(num_words = max_features)

c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb 
cat(length(x_train), "train sequences\n")
cat(length(x_test), "test sequences")
cat("Pad sequences (samples x time)\n")
x_train <- pad_sequences(x_train, maxlen = max_len)
x_test <- pad_sequences(x_test, maxlen = max_len)
cat("x_train shape:", dim(x_train), "\n")
cat("x_test shape:", dim(x_test), "\n")

# One difference between 1D and 2D convolutions is the fact that you can afford to use larger 
# convolution windows with 1D convnets. With a 2D convolution layer, a 3 × 3 convolution window 
# contains 3 * 3 = 9 feature vectors; but with a 1D convolution layer, a convolution window of size 
# 3 contains only 3 feature vectors. You can thus easily afford 1D convolution windows of size 7 or 9.

model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_features, output_dim = 128,
                  input_length = max_len) %>% 
  layer_conv_1d(filters = 32, kernel_size = 7, activation = "relu") %>% 
  layer_max_pooling_1d(pool_size = 5) %>% 
  layer_conv_1d(filters = 32, kernel_size = 7, activation = "relu") %>% 
  layer_global_max_pooling_1d() %>% 
  layer_dense(units = 1)

# validation accuracy is somewhat lower than that of the LSTM we used two sections ago, 
# but runtime is faster, both on CPU and GPU


# Combining CNNs and RNNs -------------------------------------------------------------------------

# Because 1D convnets process input patches independently, they are not sensitive to the order of the 
# timesteps (beyond a local scale, the size of the convolution windows), unlike RNNs. Of course, in 
# order to be able to recognize longer-term patterns, one could stack many convolution layers and 
# pooling layers, resulting in upper layers that would “see” long chunks of the original inputs – but 
# that’s still a fairly weak way to induce order-sensitivity. So, the can be bad are temperature 
# forecasting and good at text-tasks (IMDB).


# One strategy to combine the speed and lightness of convnets with the order-sensitivity of RNNs is 
# to use a 1D convnet as a preprocessing step before a RNN. This is especially beneficial when you’re 
# dealing with sequences that are so long they can’t realistically be processed with RNNs, such as 
# sequences with thousands of steps. The convnet will turn the long input sequence into much shorter 
# (downsampled) sequences of higher-level features. This sequence of extracted features then becomes 
# the input to the RNN part of the network.

# Because this strategy allows you to manipulate much longer sequences, you can either look at data 
# from longer ago (by increasing the lookback parameter of the data generator) or look at high-resolution
# timeseries (by decreasing the step parameter of the generator). Here, somewhat arbitrarily, you’ll use 
# a step that’s half as large, resulting in a timeseries twice as long, where the weather data is sampled 
# at a rate of 1 point per 30 minutes.

# This was previously set to 6 (one point per hour).
# Now 3 (one point per 30 min).
step <- 3 
lookback <- 720  # Unchanged
delay <- 144  # Unchanged

train_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 1,
  max_index = 200000,
  shuffle = TRUE,
  step = step
)
val_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 200001,
  max_index = 300000,
  step = step
)
test_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 300001,
  max_index = NULL,
  step = step
)
val_steps <- (300000 - 200001 - lookback) / 128
test_steps <- (nrow(data) - 300001 - lookback) / 128

model <- keras_model_sequential() %>% 
  layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu",
                input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_max_pooling_1d(pool_size = 3) %>% 
  layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu") %>% 
  layer_gru(units = 32, dropout = 0.1, recurrent_dropout = 0.5) %>% 
  layer_dense(units = 1)
summary(model)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)
history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 20,
  validation_data = val_gen,
  validation_steps = val_steps
)

