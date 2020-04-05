# Set-up, import needed modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from preprocessing import *
from mathutils import *
from plotting import *
from scipy.special import softmax

# Create a NN that can predict values from the XOR function.
plt.plot([0,1], [0,1], 'ro')
plt.plot([0,1], [1,0], 'rx')
plt.title("XOR")
plt.show()

epochs = 1000000 #50000
eprate = epochs/10
input_size, hidden_size, output_size = 2,2,1 #2,3,1
learning_rate = .1

# Data for the XOR model
X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([  [0],   [1],   [1],   [0]])

# Initialize the weights of the NN to random numbers
w_hidden = np.random.uniform(size=(input_size, hidden_size))
w_output = np.random.uniform(size=(hidden_size, output_size))

def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoid_prime(x):
	sig = sigmoid(x)
	return sig * (1-sig)

print("\n\r--- Training NN ------")

# Back-propagation algorithm:
for epoch in range(epochs):
# Fordward:
# We multiplied the matrix containing our training data with the matrix of the weights of the hidden layer. 
# Then, we applied the activation function (sigmoid) to the result and multiply that with the weight matrix of the output layer.

	act_hidden = sigmoid(np.dot(X, w_hidden))
	output = np.dot(act_hidden, w_output)
	
# The error is computed by doing simple subtraction.
	error = Y - output

	if (epoch % eprate) == 0:
		print("Epoch:", epoch)
		print(f'Error: {sum(error)}')

# Backward:
# During the backprop step, we adjust the weight matrices using the computed error and use the derivative of the sigmoid function.

	dZ = error * learning_rate
	w_output += act_hidden.T.dot(dZ)
	dH = dZ.dot(w_output.T) * sigmoid_prime(act_hidden)
	w_hidden += X.T.dot(dH)
# Error seems to be decreasing!
print("Training Completed!\n")

###############################################################
# Test NN #####################################################
print("Test NN:")
for res in range(4):
	act_hidden = sigmoid(np.dot(X[res], w_hidden))
	output = np.dot(act_hidden, w_output)
	print(X[res], ' = ', output)


