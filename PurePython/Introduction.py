# Set-up, import needed modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from preprocessing import *
from mathutils import *
from plotting import *
from scipy.special import softmax

# Prepare our environment, seed the random number generator
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize']=12,6
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Sigmoid (Activation function for the hidden layer)
x = np.linspace(-10., 10., num=100)
sig = 1/(1+np.exp(-x))
sigp = sig * (1-sig)

plt.plot(x, sig, label="sigmoid")
plt.plot(x, sigp, label="sigmoid prime")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(prop={'size':16})
plt.show()

# Hyperbolic tangent (Activation function)
x = np.linspace(-np.pi, np.pi, 100)
th = np.tanh(x)
plt.plot(x,th, label="tanh")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(prop={'size':16})
plt.show()

# RELU (Activation function)
z = np.arange(-2, 2, .1)
zero = np.zeros(len(z))
y = np.max([zero, z], axis=0)
plt.plot(z, y, label="relu")
plt.xlabel("z")
plt.legend(prop={'size':16})
plt.show()

# Softmax.
np.set_printoptions(precision=5)
x = np.array([[2, 4, 6, 8]])
m = softmax(x)
print(x)
print(m)
print(m.sum())
print("----------------------------------------------------")
print("SOFTMAX Assigns decimal probabilities to each class.")
print("The output has most of its weight corresponding to the input 8. Softmax highlights the largest values and suppresses the smaller ones.")

