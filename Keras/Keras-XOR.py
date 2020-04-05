import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

# Data for the XOR model
X = np.array([[0,0], [0,1], [1,0], [1,1]], "float32")
Y = np.array([  [0],   [1],   [1],   [0]], "float32")

# Create the model, two fully-connected layers
model = Sequential()
model.add(Dense(16, input_dim=2, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# Compile and train the NN
model.compile(loss="mean_squared_error",
              optimizer="adam",
              metrics=["binary_accuracy"])
model.fit(X, Y, nb_epoch=500, verbose=2)

# Test data
Results = model.predict(X).round()

print("\n\rTest NN:")
for i in range(4):
	print(X[i], " = ", Results[i])

