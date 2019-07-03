from keras.layers import Input, Dense, Embedding, Flatten
from keras.models import Model
from keras.utils import plot_model
import matplotlib.pyplot as plt


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