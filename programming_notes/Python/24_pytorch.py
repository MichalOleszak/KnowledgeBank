import torch


# Creating tensors in PyTorch -----------------------------------------------------------------------------------------

# Create random tensor of size 3 by 3
your_first_tensor = torch.rand(3, 3)

# Calculate the shape of the tensor
tensor_size = your_first_tensor.shape

# Create a matrix of ones with shape 3 by 3
tensor_of_ones = torch.ones(3, 3)

# Create an identity matrix with shape 3 by 3
identity_tensor = torch.eye(3)

# Element-wise multiply tensor_of_ones with identity_tensor
matrices_multiplied = torch.matmul(tensor_of_ones, identity_tensor)

# Element-wise multiply tensor_of_ones with identity_tensor
element_multiplication = tensor_of_ones * identity_tensor


# Forward propagation -------------------------------------------------------------------------------------------------

x = torch.rand(1000, 1000)
y = torch.rand(1000, 1000)
z = torch.rand(1000, 1000)

q = x + y
f = torch.matmul(z, q)

mean_f = torch.mean(f)


# Backpropagation by auto-differentiation -----------------------------------------------------------------------------

# Initialize x, y and z to values 4, -3 and 5
x = torch.tensor(4., requires_grad=True)
y = torch.tensor(-3., requires_grad=True)
z = torch.tensor(5., requires_grad=True)

# Set q to sum of x and y, set f to product of q with z
q = x + y
f = z * q

# Compute the derivatives
f.backward()

# Print the gradients
print("Gradient of x is: " + str(x.grad))
print("Gradient of y is: " + str(y.grad))
print("Gradient of z is: " + str(z.grad))


# Neural Nets ---------------------------------------------------------------------------------------------------------

# By hand
weight_1 = torch.rand(784, 1)
weight_2 = torch.rand(1, 1)
hidden_1 = torch.matmul(input_layer, weight_1)
output_layer = torch.matmul(hidden_1, weight_2)

# With PyTorch API
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Instantiate all 2 linear layers  
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
      
        # Use the instantiated layers and return x
        x = self.fc1(x)
        x = self.fc2(x)
        return x

		
# Activation functions ------------------------------------------------------------------------------------------------

# Networks with multiple layers which do not contain non-linearity (so no activation function) can be expressed 
# as neural networks with one layer. They only work with linearly separable datasets, because they are simply
# linear transformations, which can be seen from the fact that the same result can be obtained from compounding
# weights: res_1 = res_2.

hidden_1 = torch.matmul(input_layer, weight_1)
hidden_2 = torch.matmul(hidden_1, weight_2)
out_1 = torch.matmul(hidden_2, weight_3)

weight_composed_1 = torch.matmul(weight_1, weight_2)
weight = torch.matmul(weight_composed_1, weight_3)
res_2 = torch.matmul(input_layer, weight)

# RelU
relu = torch.nn.ReLU()

weight_1 = torch.rand(4, 6)
weight_2 = torch.rand(6, 2)

hidden_1 = torch.matmul(input_layer, weight_1)
hidden_1_activated = relu(hidden_1)
out_layer = torch.matmul(hidden_1_activated, weight_2)


# Loss functions ------------------------------------------------------------------------------------------------------

# Predicted scores are -1.2 for class 0 (cat), 0.12 for class 1 (car) and 4.8 
# for class 2 (frog). The ground truth is class 2 (frog). The categorical cross-entropy is:
# Initialize the scores and ground truth
logits = torch.tensor([[-1.2, 0.12, 4.8]])
ground_truth = torch.tensor([2])
loss = nn.CrossEntropyLoss(logits, ground_truth)


# Preparing a data set ------------------------------------------------------------------------------------------------
import torchvision
import torch.utils.data
from torchvision import transforms

# Transform the data to torch tensors and normalize it with mean 0.1307 and standard deviation 0.3081
transform = transforms.Compose([transforms.ToTensor(),
								transforms.Normalize((0.1307), ((0.3081)))])

# Prepare training set and testing set
trainset = torchvision.datasets.MNIST('mnist', train=True, 
									  download=True, transform=transform)
testset = torchvision.datasets.MNIST('mnist', train=False, 
									  download=True, transform=transform)

# Prepare training loader and testing loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
										 shuffle=False, num_workers=0) 


# Compute the shape of the training set and testing set
trainset_shape = trainloader.dataset.train_data.shape
testset_shape = testloader.dataset.test_data.shape

# Compute the size of the minibatch for training set and testing set
trainset_batchsize = trainloader.batch_size
testset_batchsize = trainloader.batch_size


# Training neural networks --------------------------------------------------------------------------------------------
import torch.nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):    
    	# Define all the parameters of the net
        super(Net, self).__init__()
		# Images have size 28x28x1; let's use 200 neurons in the hidden layer
        self.fc1 = nn.Linear(28 * 28 * 1, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):   
    	# Do the forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


# Instantiate the Adam optimizer and Cross-Entropy loss function
optimizer = optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()
  
for batch_idx, data_target in enumerate(train_loader):
    data = data_target[0]
    target = data_target[1]
    data = data.view(-1, 28 * 28)
    optimizer.zero_grad()

    # Complete a forward pass
    output = model(data)

    # Compute the loss, gradients and change the weights
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
	

# Using the network to make predictions -------------------------------------------------------------------------------

# Set the model in eval mode
model.eval()

total = 0
correct = 0

for i, data in enumerate(test_loader, 0):
    inputs, labels = data
    
    # Put each image into a vector
    inputs = inputs.view(-1, 28 * 28)
    
    # Do the forward pass and get the predictions
    outputs = model(inputs)
    _, outputs = torch.max(outputs.data, 1)
    total += labels.size(0)
	
    correct += (outputs == labels).sum().item()
print('The testing set accuracy of the network is: %d %%' % (100 * correct / total))


# Convolutions --------------------------------------------------------------------------------------------------------

# Convolution operator - OOP way ----------------------------------------------

# Create 10 random images of shape (1, 28, 28)
images = torch.rand(10, 1, 28, 28)

# Build 6 conv. filters of size 3x3
conv_filters = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3,
               stride=1, padding=1)

# Convolve the image with the filters 
output_feature = conv_filters(images)

# Convolution operator - Functional way ---------------------------------------

# Create 10 random images
image = torch.rand(10, 1, 28, 28)

# Create 6 filters
filters = torch.rand(6, 1, 3, 3)

# Convolve the image with the filters
output_feature = F.conv2d(image, filters, stride=1, padding=1)


# Pooling ---------------------------------------------------------------------------------------------------------

# Max pooling - OOP way
max_pooling = torch.nn.MaxPool2d(2)
output_feature = max_pooling(im)

# Max pooling - Functional way
output_feature_F = F.max_pool2d(im, 2)

# Average pooling - OOP way
avg_pooling = torch.nn.AvgPool2d(2)
output_feature = avg_pooling(im)

# Average pooling - Functional way
output_feature_F = F.avg_pool2d(im, 2)


# Building convnets ---------------------------------------------------------------------------------------------------

# Build model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Instantiate two convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1)
        
        # Instantiate the ReLU nonlinearity
        self.relu = nn.ReLU()
        
        # Instantiate a max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Instantiate a fully connected layer
        self.fc = nn.Linear(7 * 7 * 10, 10)
		
	def forward(self, x):
  
        # Apply conv followd by relu, then in next line pool
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        # Apply conv followd by relu, then in next line pool
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # Prepare the image for the fully connected layer
        x = x.view(-1, 7 * 7 * 10)

        # Apply the fully connected layer and return the result
        return self.fc(x)

# Train model
net = Net()
for i, data in enumerate(train_loader, 0):
    inputs, labels = data
    optimizer.zero_grad()

    # Compute the forward pass
    outputs = net(inputs)
        
    # Compute the loss function
    loss = criterion(outputs, labels)
        
    # Compute the gradients
    loss.backward()
    
    # Update the weights
    optimizer.step()
	
# Make predictions
for data in test_loader:
  
    # Get the image and label from data
    image, label = data
    
    # Make a forward pass in the net with your image
    output = model(image)
    
    # Argmax the results of the net
    _, predicted = torch.max(output.data, 1)
    if predicted == label:
        print("Yipes, your net made the right prediction " + str(predicted))
    else:
        print("Your net prediction was " + str(predicted) + ", but the correct label is: " + str(label))


# Sequential module ---------------------------------------------------------------------------------------------------

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Declare all the layers for feature extraction
        self.features = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1), 
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1), 
                                      nn.MaxPool2d(2, 2), nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, padding=1),
                                      nn.MaxPool2d(2, 2), nn.ReLU(inplace=True))
        # Declare all the layers for classification
        self.classifier = nn.Sequential(nn.Linear(7 * 7 * 40, 1024), nn.ReLU(inplace=True),
                                       	nn.Linear(1024, 2048), nn.ReLU(inplace=True),
                                        nn.Linear(2048, 10))

	def forward():
        # Apply the feature extractor in the input
        x = self.features(x)
        # Squeeze the three spatial dimensions in one
        x = x.view(-1, 7 * 7 * 40)
        # Classify the images
        x = self.classifier(x)
        return x
		

# Creating validation set ---------------------------------------------------------------------------------------------

# Shuffle the indices
indices = np.arange(60000)
np.random.shuffle(indices)

# Build the train loader
train_loader = torch.utils.data.DataLoader(datasets.MNIST('mnist', download=True, train=True,
                     transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
                     batch_size=64, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(indices[:55000]))

# Build the validation loader
val_loader = torch.utils.data.DataLoader(datasets.MNIST('mnist', download=True, train=True,
                   transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
                   batch_size=64, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(indices[55000:]))
				   

# Regularization ------------------------------------------------------------------------------------------------------

# L2 (weight decay)
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.001)

# Dropout
class Net(nn.Module):
    def __init__(self):
        # Define all the parameters of the net
        self.classifier = nn.Sequential(
            nn.Linear(28*28, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(200, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 10))
        
    def forward(self, x):
    	# Do the forward pass
        return self.classifier(x)
		
# Batch-normalization
# Dropout is used to regularize fully-connected layers. Batch-normalization is used to make the training 
# of convolutional neural networks more efficient, while at the same time having regularization effects.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Implement the sequential module for feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2), nn.ReLU(inplace=True), nn.BatchNorm2d(10),
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2), nn.ReLU(inplace=True), nn.BatchNorm2d(20))
        # Implement the fully connected layer for classification
        self.fc = nn.Linear(in_features=7*7*20, out_features=10)
		

# Transfer learning ---------------------------------------------------------------------------------------------------

# Use already trained model changing its last layer (different number of output classes)
model = Net()
model.load_state_dict(torch.load('my_net.pth'))
model.fc = nn.Linear(7 * 7 * 512, 26)
model.train()
train_net(model, optimizer, criterion)
print("Accuracy of the net is: " + str(model.eval()))

# Use a state-of-the-art model from torchvision library
import torchvision
model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(512, 7)