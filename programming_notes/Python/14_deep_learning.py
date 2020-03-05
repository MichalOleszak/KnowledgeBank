import numpy as np
import pandas as pd
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# Forward propagation (prediction) -------------------------------------------------------------------------------------

# Input data - values of features for a single  (there are two features)
input_data = np.array([3, 5])

# Weights - assumed to be known (there are two nodes in one hidden layer)
weights = {'node_0': np.array([2, 4]),
           'node_1': np.array([ 4, -5]),
           'output': np.array([2, 7])}

# Compute predictions
node_0_value = (input_data * weights['node_0']).sum()
node_1_value = (input_data * weights['node_1']).sum()
hidden_layer_outputs = np.array([node_0_value, node_1_value])
output = (hidden_layer_outputs * weights['output']).sum()


# Activation functions -------------------------------------------------------------------------------------------------
# An activation function is a function applied at each node. It converts the node's input into some output.
# The rectified linear activation function (ReLU) has been shown to lead to very high-performance networks. This
# function takes a single number as an input, returning 0 if it is negative, and the input itself if it is positive.

def relu(input):
    output = max(input, 0)
    return (output)


node_0_input = (input_data * weights['node_0']).sum()
node_0_output = relu(node_0_input)
node_1_input = (input_data * weights['node_1']).sum()
node_1_output = relu(node_1_input)
hidden_layer_outputs = np.array([node_0_output, node_1_output])
model_output = (hidden_layer_outputs * weights['output']).sum()


# Backward propagation -------------------------------------------------------------------------------------------------

input_data = np.array([1, 2, 3])
weights = np.array([0, 2, 1])
target = 0

# Calculating the slope
# When plotting the mean-squared error loss function against predictions, the slope is
# 2 * x * (y-xb), or 2 * input_data * error. Note that x and b may have multiple numbers (x is a vector for each data
# point, and b is a vector). In this case, the output will also be a vector,
preds = (weights * input_data).sum()
error = preds - target
slope = 2 * input_data * error

# Improving model weights
learning_rate = 0.01
weights_updated = weights - (learning_rate * slope)
preds_updated = (weights_updated * input_data).sum()
error_updated = preds_updated - target
print(error)
print(error_updated)

# Backpropagation
# - estimate the slope of the loss function w.r.t each weight
# - gradient for a weight is the product of:
#   1. node value feeding into that weight
#   2. slope of activation function for the node being fed into (0 or 1 for ReLu)
#   3. slope of loss function w.r.t output node
# - stochastic gradient descent
#   - calculate slopes on only a subset (batch) of data
#   - use a different batch of data to calculate the next update
#   - when all data is used (1 epoch finished), start over from the beginning


# Building deep learning models with keras -----------------------------------------------------------------------------

# Basic regression model
hourly_wages = pd.read_csv('data/hourly_wages.csv')
predictors = hourly_wages.drop(['wage_per_hour'], axis=1).as_matrix()
target = hourly_wages.wage_per_hour

n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(predictors, target)

# Basic classification model
titanic = pd.read_csv('data/titanic_all_numeric.csv')
predictors = titanic.drop(['survived'], axis=1).as_matrix()
target = to_categorical(titanic.survived)

n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(predictors, target)

# Calculating predictions
predictions = model.predict(pred_data)
predicted_prob_true = predictions[:, 1]


# Fine-tuning keras models ---------------------------------------------------------------------------------------------

# Changing optimization parameters
my_optimizer = SGD(lr=0.01)
model.compile(optimizer=my_optimizer, loss='categorical_crossentropy')

# Evaluating model accuracy on validation dataset
hist = model.fit(predictors, target, validation_split=0.3)

# Early stopping: Optimizing the optimization
# patiens = allowed number of epochs without improvements; one can increase the max number of epochs now
early_stopping_monitor = EarlyStopping(patience=2)
model.fit(predictors, target, epochs=30, validation_split=0.3, callbacks=[early_stopping_monitor])



