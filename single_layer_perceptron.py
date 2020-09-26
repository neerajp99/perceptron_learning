import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

data = pd.read_csv('data-2.txt', sep="\t", header=None)
x = data.drop(data.columns[2], axis=1)
y = data[2]
y.head()
# Convert Class-1 and Class-2 into 0's and 1's
for i, val in enumerate(y):
    if val == "Class-1":
        y[i] = 0
    else:
        y[i] = 1

# Sigmoid activation function
def activation_func(value):
    return (1 / (1 + np.exp(-value)))

# Function to return mea squared error
def mse(y_predicted, Y_test):
    return np.sum((y_predicted - Y_test) ** 2)

# Derivative of sigmoid function
def sigmoid_derivative(val):
    return activation_func(val) * (1 - activation_func(val))

# Model to train the perceptron using delta rule
def perceptron_train(x, y, alpha, iterations):
    # converting into numpy
    X = np.array(x)
    # dtype is a numpy.object, converting the array into astype(float)
    # otherwise it will show a message saying numpy.float64 has no attribute log10
    X = X.astype(float) 
    n = X.shape[0]
    Y = np.array(y).reshape(n, 1)
    Y = Y.astype(float) 
    
    # Eandomly initialising random weights 
    # weights = [L-1, L] where L-1 => no of neurons in prev layer, L => neurons in current neurons
    weights = np.random.random((X.shape[1], 1))
#     weights = np.random.random((X.shape[1], X.shape[1]-1))
    weights = weights.reshape(2, 1)

    # Initialise bias with random bias
    bias = np.ones((1, 1))
    
    # Final arr to include all errors
    loss_arr = []
    
    # Looping over through for N iterations
    for i in range(iterations):
        # Calcuating the predicted values of Y
        z = np.dot(X, weights) + bias
        y_predicted = activation_func(z)
        
        # Calucating mean squared loss
        loss_j = mse(y_predicted, Y)
        
        print('----- ', i, ' ----- || ', loss_j)
        # Delta rule implementation
        weights = weights - alpha * np.dot(np.transpose(X), (sigmoid_derivative(y_predicted) * (y_predicted - Y))) 
       
        # Updating the bias
        bias = bias - alpha * (sigmoid_derivative(y_predicted) * (y_predicted - Y))
        # Append the loss error to the final array
        loss_arr.append(loss_j) 
    return (weights, loss_arr)

weights, loss_arr = perceptron_train(x, y, 0.005, 1000)

# Plotting the graph for the loss J decreases over time
x_arr = np.arange(0,1000,1)
loss_arr = np.array(loss_arr)
plt.plot(x_arr, loss_arr)
        