import tensorflow as tf
import numpy as np

# Data for the XOR model
X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([  [0],   [1],   [1],   [0]])

# Placeholders
phX = tf.placeholder(tf.float32, shape=[4,2])
phY = tf.placeholder(tf.float32, shape=[4,1])

# Weights
w1 = tf.Variable([[1.0, 0.0], [1.0, 0.0]], shape=[2,2])
w2 = tf.Variable([[0.0], [1.0]], shape=[2,1])

# Biasses
b1 = tf.Variable([0.0, 0.0], shape=[2])
b2 = tf.Variable([0.0], shape=1)

# Layers
hidden = tf.sigmoid(tf.matmul(phX, w1) + b1)
#hidden = tf.math.tanh(tf.matmul(phX, w1) + b1)
output = tf.sigmoid(tf.matmul(hidden, w2) + b2)

# Error
lr = 0.1
err = tf.reduce_mean(tf.squared_difference(phY, output))
train = tf.train.GradientDescentOptimizer(lr).minimize(err)

# Prepare session and run the model
epochs = 100000
eprate = epochs/10
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

print("\n" * 1000)
print("XOR Model, using Tensorflow ", tf.__version__)
for i in range(epochs+1):
	error = sess.run(train, feed_dict={phX:X, phY:Y})
	if (i%eprate) == 0:
		print("Epoch: ", i)
		print("Error: ", sess.run(err, feed_dict={phX:X, phY:Y}))
print("Training Completed!\n")

# Test the NN
print("Test NN:")
Results = sess.run(output, feed_dict={phX:X, phY:Y})

for res in range(4):
	print(X[res], ' = ', Results[res])

sess.close()


