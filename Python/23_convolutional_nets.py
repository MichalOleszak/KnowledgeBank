# Using one-hot encoding to represent images --------------------------------------------------------------------------

# The number of image categories
n_categories = 3
# The unique values of categories in the data
categories = np.array(["shirt", "dress", "shoe"])
# Initialize ohe_labels as all zeros
ohe_labels = np.zeros((len(labels), n_categories))
# Loop over the labels
for ii in range(len(labels)):
    # Find the location of this label in the categories variable
    jj = np.where(categories == labels[ii])
    # Set the corresponding zero to one
    ohe_labels[ii, jj] = 1
# Calculate the number of correct predictions
number_correct = (test_labels * predictions).sum()
print(number_correct)
# Calculate the proportion of correct predictions
proportion_correct = number_correct / predictions.shape[0]
print(proportion_correct)


# One dimensional convolutions ----------------------------------------------------------------------------------------

# A convolution of an one-dimensional array with a kernel comprises of taking the kernel, sliding it along the array, 
# multiplying it with the items in the array that overlap with the kernel in that location and summing this product.

array = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
kernel = np.array([1, -1, 0])
conv = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

for ii in range(8):
    conv[ii] = (kernel * array[ii:ii+3]).sum()


# Image convolutions --------------------------------------------------------------------------------------------------

# The convolution of an image with a kernel summarizes a part of the image as the sum of the multiplication of that 
# part of the image with the kernel. 


kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
result = np.zeros(im.shape)

for ii in range(im.shape[0] - 3):
    for jj in range(im.shape[1] - 3):
        result[ii, jj] = (im[ii:ii+3, jj:jj+3] * kernel).sum()


# Kernels -------------------------------------------------------------------------------------------------------------

# Kernel that finds a change from low to high values in a vector:
np.array([-1, 1])

# If a vector has the opposite change, the convolution will yield negative value.

# The following kernel finds a vertical line in images:
np.array([[-1, 1, -1], 
          [-1, 1, -1], 
          [-1, 1, -1]])

# A kernel that finds a light spot surrounded by dark pixels:
kernel = np.array([[-1, -1, -1], 
                   [-1, 1, -1],
                   [-1, -1, -1]])

# A kernel that finds a dark spot surrounded by bright pixels:
kernel = np.array([[1, 1, 1], 
                   [1, -1, 1],
                   [1, 1, 1]])

# During training, kernel values are being learnt.


# Tweaking convolutions: padding --------------------------------------------------------------------------------------

# Padding allows a convolutional layer to retain the resolution of the input into this layer. This is done by adding 
# zeros around the edges of the input image, so that the convolution kernel can overlap with the pixels on the edge 
# of the image.

# Default, no padding:
model.add(Conv2D(10, kernel_size=3, activation='relu', 
                 input_shape=(img_rows, img_cols, 1), 
                 padding='valid'))
# Zero-padding:
model.add(Conv2D(10, kernel_size=3, activation='relu', 
                 input_shape=(img_rows, img_cols, 1), 
                 padding='same'))


# Tweaking convolutions: stridea --------------------------------------------------------------------------------------

# The size of the strides of the convolution kernel determines whether the kernel will skip over some of the pixels 
# as it slides along the image. This affects the size of the output because when strides are larger than one, 
# the kernel will be centered on only some of the pixels.

model.add(Conv2D(10, kernel_size=3, activation='relu', 
              input_shape=(img_rows, img_cols, 1), 
              strides=2))


# Output size of a conv layer can be calculated as follows:
# O = ((I − K + 2P)/S) + 1
# where:
# O - output size
# I - input size
# K - kernel size
# P - padding size
# S - stride


# Pooling -------------------------------------------------------------------------------------------------------------

# CNNs can have a lot of parameters. Pooling layers are often added between the convolutional layers of a neural 
# network to summarize their outputs in a condensed manner, and reduce the number of parameters in the next layer 
# in the network. This can help us if we want to train the network more rapidly, or if we don't have enough data 
# to learn a very large number of parameters.
# A pooling layer can be described as a particular kind of convolution. For every window in the input it finds 
# the maximal pixel value and passes only this pixel through.

# Result placeholder
result = np.zeros((im.shape[0]//2, im.shape[1]//2))

# Pooling operation
for ii in range(result.shape[0]):
    for jj in range(result.shape[1]):
        result[ii, jj] = np.max(im[ii*2:ii*2+2, jj*2:jj*2+2])
		

# Tracking model's learning -------------------------------------------------------------------------------------------

# Plot learning curves
training = model.fit(train_data, train_labels, epochs=3, batch_size=10, validation_split=0.2)
history = training.history
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.show()

# Using stored weights to predict in a test set
# Model weights stored in an hdf5 file can be reused to populate an untrained model. Once the weights are loaded 
# into this model, it behaves just like a model that has been trained to reach these weights. 
model.load_weights('weights.hdf5')
model.predict(test_data) 


# Regularization ------------------------------------------------------------------------------------------------------

# Dropout is a form of regularization that removes a different random subset of the units in a layer in each 
# round of training. In keras, dropout layer affects the preceding layer, so it should follow a conv layer.
model.add(Dropout(0.2))

# Batch normalization is another form of regularization that rescales the outputs of a layer to make sure that they 
# have mean 0 and standard deviation 1.
model.add(BatchNormalization())


# Interpreting the model ----------------------------------------------------------------------------------------------

# One way to interpret models is to examine the properties of the kernels in the convolutional layers.
# Load the weights into the model
model.load_weights('weights.hdf5')
# Get the first convolutional layer from the model
c1 = model.layers[0]
# Get the weights of the first convolutional layer
weights1 = c1.get_weights()
# Pull out the first channel of the first kernel in the first layer
# shape = [kernel_shape_rows, kernel_shape_cols, num_channels, num_kernels]
kernel = weights1[0][:, :, 0, 0]

# Visualizing kernel responses
# One of the ways to interpret the weights of a neural network is to see how the kernels stored in these weights 
# "see" the world. That is, what properties of an image are emphasized by this kernel.
# Convolve with the fourth image in test_data
out = convolution(test_data[3, :, :, 0], kernel)
# Visualize the result
plt.imshow(out)
plt.show()
