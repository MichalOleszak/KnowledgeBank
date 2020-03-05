library(keras)

# *** INTRO *** ----

# Convolutional Neural Networks -----------------------------------------------
# - convolutions operate on feature maps - 3D tensors of shape (height, width, 
#   depth/channel); depth can be 3 for RGB image or 1 for MNIST (gray scale)
# - slide small 2D windows over the feature map, stop at each location, extract
#   a patch of features, transform it with the same weighting matrix called
#   convolution kernel into a 1D vector of shape (depth), reassemble all of
#   this vector back into 3D output map
# - paramteres:
#    * window size: typically 3x3 or 5x5
#    * depth of output map: the number of filters computed by the convolution


# Border effects & padding, strides -------------------------------------------
# - widht and height of output map may differ from that of input map
# - border effects: for 5x5 map and 3x3 window, there are only 9 possible
#   window locations, hence output map will be 3x3
# - padding: add appropriate number of rows and cols at each side of the input map,
#   to make it possible to fit windows around every input tile;
#   argument padding = c("valid", "same"), valid = no padding, same = same sizes of
#   input and output
# - strides: move window by more than one step (strided convolution)


# Max-pooling operation -------------------------------------------------------
# - used to downsample feature maps (decrease their size in subsequent layers)
# - consists of extracting windows from the input maps and output the max value
#   of each channel: similar to convolution, but instead of transforming local 
#   patches via a learned linear transformation (kernel), it uses hardcoded max
# - usually done with 2x2 windows and stride 2, to downsample maps by factor 2
# - why?
#    * to reduce number of parameters in the model
#    * to induce hierarchies by makeing successive convolution layers look at
#      increasingly large windows


# MNIST convnets example ------------------------------------------------------
mnist <- dataset_mnist()
c(c(train_images, train_labels), c(test_images, test_labels)) %<-% mnist
train_images <- array_reshape(train_images, c(60000, 28, 28, 1))
train_images <- train_images / 255
test_images <- array_reshape(test_images, c(10000, 28, 28, 1))
test_images <- test_images / 255
train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                input_shape = c(28, 28, 1)) %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dense(units = 10, activation = 'softmax')

model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = "accuracy"
)

model %>% fit(train_images, train_labels, epochs = 5, batch_size = 64)

(results <- model %>% evaluate(test_images, test_labels))



# *** TRAINING A CONVNET FROM SCRATCH *** ----

# Get and prep cats vs dogs data (only needed once, for original kaggle data)
# source("prep_cats_and_dogs_data.R")
# prep_cats_and_dogs_data()

# Build the model -------------------------------------------------------------
model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(150, 150, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten() %>% 
  layer_dense(units = 512, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("acc")
)

# Data preprocessing: image data generator ------------------------------------
# Currently, our data sits on a drive as JPEG files, so the steps for getting 
# it into our network are roughly:
#  - Read the picture files.
#  - Decode the JPEG content to RBG grids of pixels.
#  - Convert these into floating point tensors.
#  - Rescale the pixel values (between 0 and 255) to the [0, 1] interval 
# image_data_generator() function, which can automatically turn image files 
# on disk into batches of pre-processed tensors

# Set directories
train_dir <- "data/cats_and_dogs_small/train"
validation_dir <- "data/cats_and_dogs_small/validation"

# All images will be rescaled by 1/255
train_datagen <- image_data_generator(rescale = 1/255)
validation_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  train_dir,
  train_datagen,
  target_size = c(150, 150), # All images will be resized to 150x150 (arbitrary)
  batch_size = 20,
  class_mode = "binary"  # Since we use binary_crossentropy loss, we need binary labels
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

# The generator yields these batches indefinitely: it loops endlessly over the images 
# in the target folder.
batch <- generator_next(train_generator)
str(batch)

# Fit the model to the data using the generator -------------------------------
# 
# Because the data is being generated endlessly, the generator needs to know how many 
# samples to draw from the generator before declaring an epoch over. This is the role of 
# the steps_per_epoch argument: after having drawn steps_per_epoch batches from the 
# generator – that is, after having run for steps_per_epoch gradient descent steps – the 
# fitting process will go to the next epoch. 
# 
# In this case, batches are 20-samples large, so it will take 100 batches until you 
# see your target of 2,000 samples.
history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 10,
  validation_data = validation_generator,
  validation_steps = 50
)
model %>% save_model_hdf5("cats_and_dogs_small_1.h5")
#model <- load_model_hdf5("cats_and_dogs_small_1.h5")
plot(history) # Overfitting , accuracy ~ 71%-75%

# Using data augmentation -----------------------------------------------------
#
# Data augmentation takes the approach of generating more training data from existing 
# training samples, by “augmenting” the samples via a number of random transformations 
# that yield believable-looking images. The goal is that at training time, our model 
# would never see the exact same picture twice.
#
# In Keras, this can be done by configuring a number of random transformations to be 
# performed on the images read by an image_data_generator()
datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"     # strategy for newly created pixels 
)

# Displaying some augmented images ---------------------------------------------
# We pick one image to "augment"
fnames <- list.files("data/cats_and_dogs_small/train/cats", full.names = TRUE)
img_path <- fnames[[3]]
# Convert it to an array with shape (150, 150, 3)
img <- image_load(img_path, target_size = c(150, 150))
img_array <- image_to_array(img)
img_array <- array_reshape(img_array, c(1, 150, 150, 3))
# Generator that will flow augmented images
augmentation_generator <- flow_images_from_data(
  img_array, 
  generator = datagen, 
  batch_size = 1 
)
# Plot the first 4 augmented images
op <- par(mfrow = c(2, 2), pty = "s", mar = c(1, 0, 1, 0))
for (i in 1:4) {
  batch <- generator_next(augmentation_generator)
  plot(as.raster(batch[1,,,]))
}
par(op)

# Building a model with dropout -----------------------------------------------
#
# If we train a new network using this data augmentation configuration, our network 
# will never see twice the same input. However, the inputs that it sees are still 
# heavily intercorrelated, since they come from a small number of original images – we 
# cannot produce new information, we can only remix existing information. As such, 
# this might not be quite enough to completely get rid of overfitting. To further 
# fight overfitting, we will also add a dropout layer to our model, right before the
# densely-connected classifier.
model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(150, 150, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten() %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 512, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")  

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("acc")
)

datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE
)

test_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(150, 150),
  batch_size = 32,
  class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 32,
  class_mode = "binary"
)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 100,
  validation_data = validation_generator,
  validation_steps = 50
)

model %>% save_model_hdf5("cats_and_dogs_small_2.h5")
plot(history) # better, accuracy ~ 82%


# *** USING PRETRAINED NETWORK *** ---- 

# Feature extraction ----------------------------------------------------------
#
# Feature extraction consists of using the representations learned by a previous 
# network to extract interesting features from new samples. These features are then 
# run through a new classifier, which is trained from scratch.
#
# In the case of convnets, “feature extraction” will simply consist of taking the 
# convolutional base of a previously-trained network, running the new data through it, 
# and training a new classifier on top of the output.

# Instantiate the VGG16 model:
conv_base <- application_vgg16(
  weights = "imagenet", 
  include_top = FALSE,      # whether to include the densely connected classifier
  input_shape = c(150, 150, 3)  # optional
)

# Two possible ways to go now:
# 1. Run the convbase over the data once, save result to an array and use it as input
#    to a standalone, densely conntected classifier: fast & cheap but no data 
#    augmentation possible
# 2. Extend the convbase by adding dense layers on top and run the whole thing on
#    the data: allows to use data augmentation, but very expensive

# 1. Fast feature extraction without data augmentation 
base_dir <- "data/cats_and_dogs_small"
train_dir <- file.path(base_dir, "train")
validation_dir <- file.path(base_dir, "validation")
test_dir <- file.path(base_dir, "test")
datagen <- image_data_generator(rescale = 1/255)
batch_size <- 20

extract_features <- function(directory, sample_count) {
  features <- array(0, dim = c(sample_count, 4, 4, 512))  
  labels <- array(0, dim = c(sample_count))
  generator <- flow_images_from_directory(
    directory = directory,
    generator = datagen,
    target_size = c(150, 150),
    batch_size = batch_size,
    class_mode = "binary"
  )
  i <- 0
  while(TRUE) {
    batch <- generator_next(generator)
    inputs_batch <- batch[[1]]
    labels_batch <- batch[[2]]
    features_batch <- conv_base %>% predict(inputs_batch)
    index_range <- ((i * batch_size)+1):((i + 1) * batch_size)
    features[index_range,,,] <- features_batch
    labels[index_range] <- labels_batch
    i <- i + 1
    if (i * batch_size >= sample_count)
      # Note that because generators yield data indefinitely in a loop, 
      # you must break after every image has been seen once.
      break
  }
  list(
    features = features, 
    labels = labels
  )
}

train <- extract_features(train_dir, 2000)
validation <- extract_features(validation_dir, 1000)
test <- extract_features(test_dir, 1000)

# The extracted features are currently of shape (samples, 4, 4, 512). We will feed 
# them to a densely-connected classifier, so first we must flatten them to 
# (samples, 8192).
reshape_features <- function(features) {
  array_reshape(features, dim = c(nrow(features), 4 * 4 * 512))
}

train$features <- reshape_features(train$features)
validation$features <- reshape_features(validation$features)
test$features <- reshape_features(test$features)

# At this point, we can define our densely-connected classifier (note the use of 
# dropout for regularization), and train it on the data and labels that we just recorded.
model <- keras_model_sequential() %>% 
  layer_dense(units = 256, activation = "relu", 
              input_shape = 4 * 4 * 512) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  optimizer = optimizer_rmsprop(lr = 2e-5),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)
history <- model %>% fit(
  train$features, train$labels,
  epochs = 30,
  batch_size = 20,
  validation_data = list(validation$features, validation$labels)
)

# Training is very fast, since we only have to deal with two Dense layers.
plot(history)  
# even better, accuracy ~ 90%, but also: overfitting, despite dropout
# this is because no data augmantation possible


# 2. Feature extraction with data augmentation (GPU only!)
model <- keras_model_sequential() %>% 
  conv_base %>% 
  layer_flatten() %>% 
  layer_dense(units = 256, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

# Before you compile and train the model, it’s very important to freeze the 
# convolutional base. Freezing a layer or set of layers means preventing their 
# weights from being updated during training. If you don’t do this, then the 
# representations that were previously learned by the convolutional base will be 
# modified during training. Because the dense layers on top are randomly initialized, 
# very large weight updates would be propagated through the network, effectively
# destroying the representations previously learned.
cat("This is the number of trainable weights before freezing",
    "the conv base:", length(model$trainable_weights), "\n")
freeze_weights(conv_base)
cat("This is the number of trainable weights after freezing",
    "the conv base:", length(model$trainable_weights), "\n")
# Need recompiling the model to take effect!

train_datagen = image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)
test_datagen <- image_data_generator(rescale = 1/255) # Don't augment val data!
train_generator <- flow_images_from_directory(
  train_dir,
  train_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)
validation_generator <- flow_images_from_directory(
  validation_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 2e-5),
  metrics = c("accuracy")
)
history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 50
)

plot(history) # accuracy ~ 90%


# Fine tuning -----------------------------------------------------------------
#
# Another widely used technique for model reuse, complementary to feature extraction, 
# is fine-tuning. Fine-tuning consists in unfreezing a few of the top layers of a frozen
# model base used for feature extraction, and jointly training both the newly added part 
# of the model (in our case, the fully-connected classifier) and these top layers. 
# This is called “fine-tuning” because it slightly adjusts the more abstract 
# representations of the model being reused, in order to make them more relevant for 
# the problem at hand.
#
# We have stated before that it was necessary to freeze the convolution base of VGG16 
# in order to be able to train a randomly initialized classifier on top. For the same 
# reason, it is only possible to fine-tune the top layers of the convolutional base 
# once the classifier on top has already been trained. If the classified wasn’t already 
# trained, then the error signal propagating through the network during training would 
# be too large, and the representations previously learned by the layers being fine-tuned
# would be destroyed. Thus the steps for fine-tuning a network are as follow:
#
# 1. Add your custom network on top of an already trained base network.
# 2. Freeze the base network.
# 3. Train the part you added.
# 4. Unfreeze some layers in the base network.
# 5. Jointly train both these layers and the part you added.
#
# We have already completed the first 3 steps when doing feature extraction. 
# Let’s proceed with the 4th step: we will unfreeze our conv_base, and then freeze 
# individual layers inside of it.
unfreeze_weights(conv_base, from = "block3_conv1")

# Now we can start fine-tuning our network. We will do this with the RMSprop optimizer,
# using a very low learning rate. The reason for using a low learning rate is that we 
# want to limit the magnitude of the modifications we make to the representations of 
# the layers that we are fine-tuning. Updates that are too large may harm these 
# representations.
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-5),
  metrics = c("accuracy")
)
history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 100,
  validation_data = validation_generator,
  validation_steps = 50
)
save_model_hdf5(model, "cats_and_dogs_small_4.h5")
plot(history)   # accuracy ~ 96%!

# How could accuracy stay stable or improve if the loss isn’t decreasing? The answer is 
# simple: what you display is an average of pointwise loss values; but what matters for 
# accuracy is the distribution of the loss values, not their average, because accuracy 
# is the result of a binary thresholding of the class probability predicted by the model.
# The model may still be improving even if this isn’t reflected in the average loss.

# Evaluate the model on test data
test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)
model %>% evaluate_generator(test_generator, steps = 50)


# *** VISUALIZING WHAT CONVNETS LEARN *** ----

# Visualizing intermediate outputs --------------------------------------------
# This is useful to understand how successive convnet layers transform their input, 
#and to get a first idea of the meaning of individual convnet filters.

# Visualizing intermediate activations consists in displaying the feature maps that are 
# output by various convolution and pooling layers in a network, given a certain input
# (the output of a layer is often called its “activation”, the output of the activation
# function). This gives a view into how an input is decomposed unto the different 
# filters learned by the network. These feature maps we want to visualize have 3 
# dimensions: width, height, and depth (channels). Each channel encodes relatively 
# independent features, so the proper way to visualize these feature maps is by 
# independently plotting the contents of every channel, as a 2D image.

model <- load_model_hdf5("cats_and_dogs_small_2.h5")

# This will be the input image we will use – a picture of a cat, not part of images 
# that the network was trained on:
img_path <- "data/cats_and_dogs_small/test/cats/cat.1700.jpg"
# We preprocess the image into a 4D tensor
img <- image_load(img_path, target_size = c(150, 150))
img_tensor <- image_to_array(img)
img_tensor <- array_reshape(img_tensor, c(1, 150, 150, 3))
# Remember that the model was trained on inputs
# that were preprocessed in the following way:
img_tensor <- img_tensor / 255
dim(img_tensor)
# Display the picture
plot(as.raster(img_tensor[1,,,]))

# In order to extract the feature maps you want to look at, you’ll create a Keras model 
# that takes batches of images as input, and outputs the activations of all convolution
# and pooling layers. To do this, we will use the keras_model() function, which takes 
# two arguments: an input tensor (or list of input tensors) and an output tensor (or 
# list of output tensors)

# Extracts the outputs of the top 8 layers:
layer_outputs <- lapply(model$layers[1:8], function(layer) layer$output)
# Creates a model that will return these outputs, given the model input:
activation_model <- keras_model(inputs = model$input, outputs = layer_outputs)

# When fed an image input, this model returns the values of the layer activations in 
# the original model.

# Returns a list of five arrays: one array per layer activation
activations <- activation_model %>% predict(img_tensor)

first_layer_activation <- activations[[1]]
dim(first_layer_activation)

# Func for plotting the channel
plot_channel <- function(channel) {
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(channel), axes = FALSE, asp = 1, 
        col = terrain.colors(12))
}

# Visualize the 5th and 7th channel
plot_channel(first_layer_activation[1,,,5])
plot_channel(first_layer_activation[1,,,7])

# Extract and plot every channel in each of our 8 activation maps, and we will 
# stack the results in one big image tensor, with channels stacked side by side.
dir.create("cat_activations")
image_size <- 58
images_per_row <- 16
for (i in 1:8) {
  
  layer_activation <- activations[[i]]
  layer_name <- model$layers[[i]]$name
  
  n_features <- dim(layer_activation)[[4]]
  n_cols <- n_features %/% images_per_row
  
  png(paste0("cat_activations/", i, "_", layer_name, ".png"), 
      width = image_size * images_per_row, 
      height = image_size * n_cols)
  op <- par(mfrow = c(n_cols, images_per_row), mai = rep_len(0.02, 4))
  
  for (col in 0:(n_cols-1)) {
    for (row in 0:(images_per_row-1)) {
      channel_image <- layer_activation[1,,,(col*images_per_row) + row + 1]
      plot_channel(channel_image)
    }
  }
  
  par(op)
  dev.off()
}


# Visualizing convnets filters ------------------------------------------------
# This is useful to understand precisely what visual pattern or concept each filter 
# in a convnet is receptive to.

# Another easy thing to do to inspect the filters learned by convnets is to display 
# the visual pattern that each filter is meant to respond to. This can be done with 
# gradient ascent in input space: applying gradient descent to the value of the input 
# image of a convnet so as to maximize the response of a specific filter, starting from 
# a blank input image. The resulting input image would be one that the chosen filter is 
# maximally responsive to.
#
# The process is simple: we will build a loss function that maximizes the value of 
# a given filter in a given convolution layer, then we will use stochastic gradient
# descent to adjust the values of the input image so as to maximize this activation 
# value. For instance, here’s a loss for the activation of filter 0 in the layer 
# “block3_conv1” of the VGG16 network, pre-trained on ImageNet:
model <- application_vgg16(
  weights = "imagenet", 
  include_top = FALSE
)
layer_name <- "block3_conv1"
filter_index <- 1
layer_output <- get_layer(model, layer_name)$output
loss <- k_mean(layer_output[,,,filter_index])

# To implement gradient descent, we will need the gradient of this loss with respect 
# to the model’s input. To do this, we will use the k_gradients Keras backend function

# The call to `gradients` returns a list of tensors (of size 1 in this case)
# hence we only keep the first element -- which is a tensor.
grads <- k_gradients(loss, model$input)[[1]] 

# A non-obvious trick to use for the gradient descent process to go smoothly is to 
# normalize the gradient tensor, by dividing it by its L2 norm (the square root of 
# the average of the square of the values in the tensor). This ensures that the 
# magnitude of the updates done to the input image is always within a same range.

# We add 1e-5 before dividing so as to avoid accidentally dividing by 0.
grads <- grads / (k_sqrt(k_mean(k_square(grads))) + 1e-5)

# Now you need a way to compute the value of the loss tensor and the gradient tensor, 
# given an input image. You can define a Keras backend function to do this: iterate 
# is a function that takes a tensor (as a list of tensors of size 1) and returns
# a list of two tensors: the loss value and the gradient value.

iterate <- k_function(list(model$input), list(loss, grads))
# Let's test it
c(loss_value, grads_value) %<-%
  iterate(list(array(0, dim = c(1, 150, 150, 3))))

# At this point we can define an R loop to do stochastic gradient descent
# We start from a gray image with some noise
input_img_data <-
  array(runif(150 * 150 * 3), dim = c(1, 150, 150, 3)) * 20 + 128 
step <- 1  # this is the magnitude of each gradient update
for (i in 1:40) { 
  # Compute the loss value and gradient value
  c(loss_value, grads_value) %<-% iterate(list(input_img_data))
  # Here we adjust the input image in the direction that maximizes the loss
  input_img_data <- input_img_data + (grads_value * step)
}

# The resulting image tensor is a floating-point tensor of shape (1, 150, 150, 3), 
# with values that may not be integers within [0, 255]. Hence you need to post-process 
# this tensor to turn it into a displayable image. You do so with the following 
# straightforward utility function.
deprocess_image <- function(x) {
  
  dms <- dim(x)
  
  # normalize tensor: center on 0., ensure std is 0.1
  x <- x - mean(x) 
  x <- x / (sd(x) + 1e-5)
  x <- x * 0.1 
  
  # clip to [0, 1]
  x <- x + 0.5 
  x <- pmax(0, pmin(x, 1))
  
  # Reshape to original image dimensions
  array(x, dim = dms)
}

# Now you have all the pieces. Let’s put them together into an R function that takes 
# as input a layer name and a filter index, and returns a valid image tensor representing 
# the pattern that maximizes the activation of the specified filter.
generate_pattern <- function(layer_name, filter_index, size = 150) {
  
  # Build a loss function that maximizes the activation
  # of the nth filter of the layer considered.
  layer_output <- model$get_layer(layer_name)$output
  loss <- k_mean(layer_output[,,,filter_index]) 
  
  # Compute the gradient of the input picture wrt this loss
  grads <- k_gradients(loss, model$input)[[1]]
  
  # Normalization trick: we normalize the gradient
  grads <- grads / (k_sqrt(k_mean(k_square(grads))) + 1e-5)
  
  # This function returns the loss and grads given the input picture
  iterate <- k_function(list(model$input), list(loss, grads))
  
  # We start from a gray image with some noise
  input_img_data <- 
    array(runif(size * size * 3), dim = c(1, size, size, 3)) * 20 + 128
  
  # Run gradient ascent for 40 steps
  step <- 1
  for (i in 1:40) {
    c(loss_value, grads_value) %<-% iterate(list(input_img_data))
    input_img_data <- input_img_data + (grads_value * step) 
  }
  
  img <- input_img_data[1,,,]
  deprocess_image(img) 
}

# Plot filter 1 in block3, layer 1
library(grid)
grid.raster(generate_pattern("block3_conv1", 1))

# Plot the first 64 filters in each layer, and will only look at the first layer of 
# each convolution block (block1_conv1, block2_conv1, block3_conv1, block4_conv1, 
# block5_conv1). Arrange the outputs on a 8x8 grid of filter patterns.
library(grid)
library(gridExtra)
dir.create("vgg_filters")
for (layer_name in c("block1_conv1", "block2_conv1", 
                     "block3_conv1", "block4_conv1")) {
  size <- 140
  
  png(paste0("vgg_filters/", layer_name, ".png"),
      width = 8 * size, height = 8 * size)
  
  grobs <- list()
  for (i in 0:7) {
    for (j in 0:7) {
      pattern <- generate_pattern(layer_name, i + (j*8) + 1, size = size)
      grob <- rasterGrob(pattern, 
                         width = unit(0.9, "npc"), 
                         height = unit(0.9, "npc"))
      grobs[[length(grobs)+1]] <- grob
    }  
  }
  
  grid.arrange(grobs = grobs, ncol = 8)
  dev.off()
}


# Visualizing heatmaps of class activation in an image ------------------------
# This is useful to understand which part of an image where identified as belonging 
# to a given class, and thus allows to localize objects in images.

# This general category of techniques is called class activation map (CAM) visualization, 
# and it consists of producing heatmaps of class activation over input images. 
# A class-activation heatmap is a 2D grid of scores associated with a specific output 
# class, computed for every location in any input image, indicating how important each
# location is with respect to the class under consideration. 

# The specific implementation used here is the one described in “Grad-CAM: Visual 
# Explanations from Deep Networks via Gradient-based Localization.”.  It’s very simple: 
# it consists of taking the output feature map of a convolution layer, given an input 
# image, and weighing every channel in that feature map by the gradient of the class 
# with respect to the channel. Intuitively, one way to understand this trick is that 
# you’re weighting a spatial map of “how intensely the input image activates different 
# channels” by “how important each channel is with regard to the class,” resulting in 
# a spatial map of “how intensely the input image activates the class.”

# Clear out the session
k_clear_session()
# Note that we are including the densely-connected classifier on top;
# all previous times, we were discarding it.
model <- application_vgg16(weights = "imagenet")

# The local path to our target image
img_path <- "data/elephant.jpg"

# Start witih image of size 224 × 224
img <- image_load(img_path, target_size = c(224, 224)) %>% 
  # Array of shape (224, 224, 3)
  image_to_array() %>% 
  # Adds a dimension to transform the array into a batch of size (1, 224, 224, 3)
  array_reshape(dim = c(1, 224, 224, 3)) %>% 
  # Preprocesses the batch (this does channel-wise color normalization)
  imagenet_preprocess_input()

# You can now run the pretrained network on the image and decode its prediction 
# vector back to a human-readable format
preds <- model %>% predict(img)
imagenet_decode_predictions(preds, top = 3)[[1]]

# Set up the Grad-CAM process
# This is the "african elephant" entry in the prediction vector
african_elephant_output <- model$output[, 387]
# The is the output feature map of the `block5_conv3` layer,
# the last convolutional layer in VGG16
last_conv_layer <- model %>% get_layer("block5_conv3")
# This is the gradient of the "african elephant" class with regard to
# the output feature map of `block5_conv3`
grads <- k_gradients(african_elephant_output, last_conv_layer$output)[[1]]
# This is a vector of shape (512,), where each entry
# is the mean intensity of the gradient over a specific feature map channel
pooled_grads <- k_mean(grads, axis = c(1, 2, 3))
# This function allows us to access the values of the quantities we just defined:
# `pooled_grads` and the output feature map of `block5_conv3`,
# given a sample image
iterate <- k_function(list(model$input),
                      list(pooled_grads, last_conv_layer$output[1,,,])) 
# These are the values of these two quantities, as arrays,
# given our sample image of two elephants
c(pooled_grads_value, conv_layer_output_value) %<-% iterate(list(img))
# We multiply each channel in the feature map array
# by "how important this channel is" with regard to the elephant class
for (i in 1:512) {
  conv_layer_output_value[,,i] <- 
    conv_layer_output_value[,,i] * pooled_grads_value[[i]] 
}
# The channel-wise mean of the resulting feature map
# is our heatmap of class activation
heatmap <- apply(conv_layer_output_value, c(1,2), mean)
# Normalize the heatmap between 0 and 1
heatmap <- pmax(heatmap, 0) 
heatmap <- heatmap / max(heatmap)
write_heatmap <- function(heatmap, filename, width = 224, height = 224,
                          bg = "white", col = terrain.colors(12)) {
  png(filename, width = width, height = height, bg = bg)
  op = par(mar = c(0,0,0,0))
  on.exit({par(op); dev.off()}, add = TRUE)
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(heatmap), axes = FALSE, asp = 1, col = col)
}
write_heatmap(heatmap, "data/elephant_heatmap.png") 

# use the magick package to generate an image that superimposes the original 
# image with the heatmap just obtained
library(magick) 
library(viridis) 
# Read the original elephant image and it's geometry
image <- image_read(img_path)
info <- image_info(image) 
geometry <- sprintf("%dx%d!", info$width, info$height) 
# Create a blended / transparent version of the heatmap image
pal <- col2rgb(viridis(20), alpha = TRUE) 
alpha <- floor(seq(0, 255, length = ncol(pal))) 
pal_col <- rgb(t(pal), alpha = alpha, maxColorValue = 255)
write_heatmap(heatmap, "data/elephant_overlay.png", 
              width = 14, height = 14, bg = NA, col = pal_col) 
# Overlay the heatmap
image_read("data/elephant_overlay.png") %>% 
  image_resize(geometry, filter = "quadratic") %>% 
  image_composite(image, operator = "blend", compose_args = "20") %>%
  plot() 
