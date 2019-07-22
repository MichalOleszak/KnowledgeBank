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


# 



