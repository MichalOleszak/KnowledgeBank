# Constants and variables ---------------------------------------------------------------------------------------------

# A constant is the simplest category of tensor. It can't be trained, which makes it a bad choice for a model's 
# parameters, but a good choice for input data. Input data may be transformed after it is defined or loaded, 
# but is typically not modified by the training process.

# Define a 3x4 tensor with all values equal to 9
A34 = fill([3, 4], 9)

# Define a tensor of ones with the same shape as A34
B34 = ones_like(A34)

# Define the one-dimensional vector, C1
C1 = constant([1, 2, 3, 4])

# Print C1 as a numpy array
print(C1.numpy())

# Unlike a constant, a variable's value can be modified. This will be quite useful when we want to train 
# a model by updating its parameters. 

# Define the 1-dimensional variable A1
A1 = Variable([1, 2, 3, 4])
print(A1)

# Convert A1 to a numpy array and assign it to B1
B1 = A1.numpy()
print(B1)


# Basic operations ----------------------------------------------------------------------------------------------------

A1 = constant([1, 2, 3, 4])
A23 = constant([[1, 2, 3], [1, 6, 4]])
B1 = ones_like(A1)
B23 = ones_like(A23)

# Element-wise multiplication
C1 = A1 * B1
C1 = multiply(A1, B1)
C23 = A23 * B23
C23 = multiply(A23, B23)

# Matrix multiplication
X = constant([[1, 2], [2, 1], [5, 8], [6, 10]])
b = constant([[1], [2]])
y = constant([[6], [4], [20], [23]])
ypred = matmul(X, b)

# Summing over tensor dimensions
reduce_sum(X,)
reduce_sum(X, 0) 
reduce_sum(X, 1)


# Advanced operations -------------------------------------------------------------------------------------------------

# Reshaping tensors
image = ones([16, 16])
# Reshape image into a vector
image_vector = reshape(image, (256, 1))
# Reshape image into a higher dimensional tensor
image_tensor = reshape(image, (4, 4, 4, 4))


# Optimizing with gradients
# You are given a loss function, y=x^2, which you want to minimize. You can do this by computing the slope using 
# the GradientTape() operation at different values of x. If the slope is positive, you can decrease the loss by 
# lowering x. If it is negative, you can decrease it by increasing x. This is how gradient descent works.

def compute_gradient(x0):
    # Define x as a variable with an initial value of x0
  x = Variable(x0)
  with GradientTape() as tape:
    tape.watch(x)
        # Define y using the multiply operation
    y = x * x
    # Return the gradient of y with respect to x
  return tape.gradient(y, x).numpy()

# Compute and print gradients at x = -1, 1, and 0
print(compute_gradient(-1.0))
print(compute_gradient(1.0))
print(compute_gradient(0.0))


# Input data ----------------------------------------------------------------------------------------------------------

# Setting data type
scalar = tf.constant(0.1, tf.float32)
bedrooms = tf.cast(housing['bedrooms'], tf.float32)
product = tf.multiply(bedrooms, scalar)
# Use a numpy array to define price as a 32-bit float
price = np.array(housing['price'], np.float32)
# Define waterfront as a Boolean using cast
waterfront = tf.cast(housing['waterfront'], tf.bool)


# Loss functions ------------------------------------------------------------------------------------------------------

loss = tf.keras.losses.mae(targets, predictions)
print(loss.numpy())

# Construct a function of the trainable model variables that returns the loss. You can then repeatedly evaluate this 
# function for different variable values until you find the minimum. In practice, you will pass this function to an 
# optimizer in tensorflow.
# Initialize a variable named scalar

scalar = Variable(1.0, dtype=tf.float32)

def loss_function(scalar, features, targets):
  predictions = scalar * features
  return keras.losses.mae(target, predictions)

print(loss_function(scalar, features, target).numpy())


# Linear regression ---------------------------------------------------------------------------------------------------

intercept = Variable(0.1, float32)
slope = Variable(0.1, float32)

def loss_function(intercept, slope):
  pred_price_log = intercept + slope * lot_size_log
  return keras.losses.mse(price_log, pred_price_log)

# Initialize an adam optimizer
opt = keras.optimizers.Adam(lr=0.5)
for j in range(500):
  # Apply minimize, pass the loss function, and supply the variables
  opt.minimize(lambda: loss_function(intercept, slope), var_list=[intercept, slope])
  # Print every 100th value of the loss    
  if j % 100 == 0:
    print(loss_function(intercept, slope).numpy())


# Batch training ------------------------------------------------------------------------------------------------------

# Define the intercept and slope
intercept = Variable(10.0, float32)
slope = Variable(0.5, float32)

# Define the loss function
def loss_function(intercept, slope, features, target):
  # Define the predicted values
  predictions = intercept + slope * features
    
  # Define the MSE loss
  return keras.losses.mse(target, predictions)

# Initialize adam optimizer
opt = keras.optimizers.Adam()

# Load data in batches
for batch in pd.read_csv('kc_house_data.csv', chunksize=100):
  size_batch = np.array(batch['sqft_lot'], np.float32)
    
  # Extract the price values for the current batch
  price_batch = np.array(batch['price'], np.float32)

  # Complete the loss, fill in the variable list, and minimize
  opt.minimize(lambda: loss_function(intercept, slope, size_batch, price_batch), var_list=[intercept, slope])

# Print trained parameters
print(intercept.numpy(), slope.numpy())


# Neural Networks: Dense Layers ---------------------------------------------------------------------------------------
# There are two ways to define a dense layer in tensorflow. The first involves the use of low-level, linear algebraic 
# operations. The second makes use of high-level keras operations. 

# Low-level
weights1 = Variable(ones((3, 2)))
product1 = matmul(borrower_features, weights1)
dense1 = keras.activations.sigmoid(product1)
weights2 = Variable(ones((2, 1)))
product2 = matmul(dense1, weights2)
prediction = keras.activations.sigmoid(product2)
print('\n prediction: {}'.format(prediction.numpy()[0,0]))

# High-level
dense1 = keras.layers.Dense(7, activation='sigmoid')(borrower_features)
dense2 = keras.layers.Dense(3, activation='sigmoid')(dense1)
predictions = keras.layers.Dense(1, activation='sigmoid')(dense2)


# Neural Networks: Activation Functions -------------------------------------------------------------------------------

# Binary classification
inputs = constant(bill_amounts, float32)
dense1 = keras.layers.Dense(3, activation='relu')(inputs)
dense2 = keras.layers.Dense(2, activation='relu')(dense1)
outputs = keras.layers.Dense(1, activation='sigmoid')(dense2)
error = default[:5] - outputs.numpy()[:5]
print(error)

# Multiclass classification
inputs = constant(borrower_features, float32)
dense1 = keras.layers.Dense(10, activation='sigmoid')(inputs)
dense2 = keras.layers.Dense(8, activation='relu')(dense1)
outputs = keras.layers.Dense(6, activation='softmax')(dense2)
print(outputs.numpy()[:5])


# Neural Networks: Optimizers ----------------------------------------------------------------------------------------

# Minimize loss with two different initial starting values:
# Initialize x_1 and x_2
x_1 = Variable(6.0,float32)
x_2 = Variable(0.3,float32)

# Define the optimization operation
opt = keras.optimizers.SGD(learning_rate=0.01)
for j in range(100):
  # Perform minimization using the loss function and x_1
  opt.minimize(lambda: loss_function(x_1), var_list=[x_1])
  # Perform minimization using the loss function and x_2
  opt.minimize(lambda: loss_function(x_2), var_list=[x_2])

# Momentum allows the optimizer to break through local minima:
# Initialize x_1 and x_2
x_1 = Variable(0.05,float32)
x_2 = Variable(0.05,float32)

opt_1 = keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.99)
opt_2 = keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.0)

for j in range(100):
  opt_1.minimize(lambda: loss_function(x_1), var_list=[x_1])
    # Define the minimization operation for opt_2
  opt_2.minimize(lambda: loss_function(x_2), var_list=[x_2]) 


# Neural Networks: Training -------------------------------------------------------------------------------------------

# A good initialization can reduce the amount of time needed to find the global minimum.
weights1 = Variable(random.normal([23, 7]))
bias1 = Variable(ones([7]))
weights2 = Variable(random.normal([7, 1]))
bias2 = Variable(0.0)

def loss_function(weights1, bias1, weights2, bias2, features, targets):
  layer1 = nn.relu(matmul(features, weights1) + bias1)
  dropout = keras.layers.Dropout(0.25)(layer1)
  layer2 = nn.sigmoid(matmul(dropout, weights2) + bias2)
  return keras.losses.binary_crossentropy(targets, layer2)
  
for j in range(0, 30000, 2000):
  features, targets = borrower_features[j:j+2000, :], default[j:j+2000, :]
  opt.minimize(lambda: loss_function(weights1, bias1, weights2, bias2, features, targets), 
    var_list=[weights1, bias1, weights2, bias2])







