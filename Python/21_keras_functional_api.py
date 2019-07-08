from keras.layers import Input, Dense, Embedding, Flatten, Subtract
from keras.models import Model
from keras.utils import plot_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid


# The Keras Functional API --------------------------------------------------------------------------------------------

# A layer is a function that has tensors as input and output
input_tensor = Input(shape=(1,))
# one way
output_layer = Dense(1)
output_tensor = output_layer(input_tensor)
# another way; same as above
output_tensor = Dense(1)(input_tensor)

# This is linear regression model with 2 params: intercept and slope (or weight and bias)
model = Model(input_tensor, output_tensor)
model.compile(optimizer='adam', loss='mean_absolute_error')

# Summarize the model
model.summary()

# Plot the model
plot_model(model, to_file='model.png')

# Display the image
data = plt.imread('model.png')
plt.imshow(data)
plt.show()


# Two Input Networks Using Categorical Embeddings, Shared Layers, and Merge Layers ------------------------------------

# Embeddings
# Shared layers allow a model to use the same weight matrix for multiple steps. Create an embedding layer that maps 
# each team ID to a single number representing that team's strength. The embedding layer is a lot like a dictionary, 
# but your model learns the values for each key.
team_lookup = Embedding(input_dim=n_teams,
                        output_dim=1,
                        input_length=1,
                        name='Team-Strength')


# Create an input layer for the team ID
teamid_in = Input(shape=(1,))
# Lookup the input in the team strength embedding layer
strength_lookup = team_lookup(teamid_in)
# Flatten the output
strength_lookup_flat = Flatten()(strength_lookup)
# Combine the operations into a single, re-usable model
team_strength_model = Model(teamid_in, strength_lookup_flat, name='Team-Strength-Model')

# Shared layers
# In this dataset, you have 10,888 unique teams. You want to learn a strength rating for each team, such that if 
# any pair of teams plays each other, you can predict the score, even if those two teams have never played before. 
# Furthermore, you want the strength rating to be the same, regardless of whether the team is the home team or the 
# away team.

team_in_1 = Input(shape=(1,), name='Team-1-In')
team_in_2 = Input(shape=(1,), name='Team-2-In')
team_1_strength = team_strength_model(team_in_1)
team_2_strength = team_strength_model(team_in_2)


# Merge layers
# Subtract the team strengths to determine which team is expected to win the game. The subtract layer will combine 
# the weights from the two layers by subtracting them.

score_diff = Subtract()([team_1_strength, team_2_strength])
model = Model([team_in_1, team_in_2], score_diff)
model.compile('adam', 'mean_absolute_error')

# Fit the model to the regular season training data
input_1 = games_season['team_1']
input_2 = games_season['team_2']
model.fit([input_1, input_2],
          games_season['score_diff'],
          epochs=1,
          batch_size=2048,
          validation_split=0.1,
          verbose=True)
 
# Evaluate the model on the tournament test data
input_1 = games_tourney['team_1']
input_2 = games_tourney['team_2']
model.evaluate([input_1, input_2], games_tourney['score_diff'])


# Multiple-input models -----------------------------------------------------------------------------------------------
# This model will have three inputs: team_id_1, team_id_2, and home. The team IDs will be integers that you look up 
# in your team strength model from the previous chapter, and home will be a binary variable, 1 if team_1 is playing 
# at home, 0 if they are not.

# Create inputs
team_in_1 = Input(shape=(1,), name='Team-1-In')
team_in_2 = Input(shape=(1,), name='Team-2-In')
home_in = Input(shape=(1,), name='Home-In')

# Lookup the team inputs in the team strength model
team_1_strength = team_strength_model(team_in_1)
team_2_strength = team_strength_model(team_in_2)

# Combine the team strengths with the home input using a Concatenate layer, then add a Dense layer
out = Concatenate()([team_1_strength, team_2_strength, home_in])
out = Dense(1)(out)

# Make and compile a model
model = Model([team_in_1, team_in_2, home_in], out)
model.compile(optimizer='adam', loss='mean_absolute_error')

# Fit the model and evaluate
model.fit([games_season['team_1'], games_season['team_2'], games_season['home']],
          games_season['score_diff'],
          epochs=1,
          verbose=True,
          validation_split=0.1,
          batch_size=2048)

# Evaluate the model on the games_tourney dataset
model.evaluate([games_tourney['team_1'], games_tourney['team_2'], games_tourney['home']],
games_tourney['score_diff'])


# Summarizing and plotting models
model.summary()

plot_model(model, to_file='model.png')
data = plt.imread('model.png')
plt.imshow(data)
plt.show()

# Stacking models
# You'll use the prediction from the regular season model as an input to the tournament model. 
# This is a form of "model stacking":
# Regular season data -> regular season model -> predict tournament data -> tournament predictions 
# + tournament data -> tournament model
games_tourney['pred'] = model.predict([games_tourney['team_1'], 
									   games_tourney['team_2'], 
									   games_tourney['home']])

# Input layer with multiple columns: a different way to create models with multiple inputs. 
# This method only works for purely numeric data, but its a much simpler approach to making 
# multi-variate neural networks.
input_tensor = Input((3,))
output_tensor = Dense(1)(input_tensor)
model = Model(input_tensor, output_tensor)
model.compile(optimizer='adam', loss='mean_absolute_error')
model.fit(games_tourney_train[['home', 'seed_diff', 'pred']],
          games_tourney_train['score_diff'],
          epochs=1,
          verbose=True)
model.evaluate(games_tourney_test[['home', 'seed_diff', 'prediction']], 
               games_tourney_test[['score_diff']])


# Multiple Outputs ----------------------------------------------------------------------------------------------------
# Use the tournament data to build one model that makes two predictions: the scores of both teams in a given game. 
# The inputs will be the seed difference of the two teams, as well as the predicted score difference from the model
# built earlier. The output will be the predicted score for team 1 as well as team 2. This is called "multiple target 
# regression": one model making more than one prediction.

input_tensor = Input((2,))
output_tensor = Dense(2)(input_tensor)
model = Model(input_tensor, output_tensor)
model.compile('adam', 'mean_absolute_error')
model.fit(games_tourney_train[['seed_diff', 'pred']],
  		  games_tourney_train[['score_1', 'score_2']],
  		  verbose=True,
  		  epochs=100,
  		  batch_size=16384)

# Both output weights are about 72, as on average a team will score about 72 points in the tournament
print(model.get_weights())
print(games_tourney_train.mean(axis=0))

model.evaluate(games_tourney_test[['seed_diff', 'pred']],
			   games_tourney_test[['score_1', 'score_2']])


# Single model for classification and regression ----------------------------------------------------------------------
# Predict the score difference, instead of both team's scores and then predict the probability that team 1 won.
# In this model, turn off the bias, or intercept for each layer. Your inputs (seed difference and predicted score 
# difference) have a mean of very close to zero, and your outputs both have means that are close to zero, so your model 
# shouldn't need the bias term to fit the data well.

input_tensor = Input((2,))
output_tensor_1 = Dense(1, activation='linear', use_bias=False)(input_tensor)
output_tensor_2 = Dense(1, activation='sigmoid', use_bias=False)(output_tensor_1)
model = Model(input_tensor, [output_tensor_1, output_tensor_2])
model.compile(loss=['mean_absolute_error', 'binary_crossentropy'], optimizer=Adam(lr=0.01))
model.fit(games_tourney_train[['seed_diff', 'pred']],
          [games_tourney_train[['score_diff']], games_tourney_train[['won']]],
          epochs=10,
          verbose=True,
          batch_size=16384)

# Now you should take a look at the weights for this model. In particular, note the last weight of the model. 
# This weight converts the predicted score difference to a predicted win probability. If you multiply the predicted 
# score difference by the last weight of the model and then apply the sigmoid function, you get the win probability 
# of the game.

print(model.get_weights())
print(games_tourney_train.mean(axis=0))

# Weight from the model
weight = 0.14
# Print the approximate win probability predicted close game
print(sigmoid(1 * weight))
# Print the approximate win probability predicted blowout game
print(sigmoid(10 * weight))




