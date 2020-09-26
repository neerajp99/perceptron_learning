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

# Custom evaluate function to get the predicted value of Y using the sigmoid activation
def evaluate(theta, X_test):
    # Initialise it with random bias
    bias = np.ones((1, 1))
    y_predicted = activation_func(np.dot(X_test, theta) + bias)
    for i, val in enumerate(y_predicted):
        if val >= 0.5:
            y_predicted[i] = 1
        else:
            y_predicted[i] = 0
    return y_predicted

# Import data and create a 2-D matrix out of it
data = open('data-2.txt', 'r') 
Lines = data.readlines()
initial_data = []
for i, line in enumerate(Lines):
    initial_data.append(Lines[i].strip().split('\t'))
# Updating Class-1 and Class-2 into 0's and 1's
for i in initial_data:
    if i[2] == 'Class-1':
        i[2] = 0
    else:
        i[2] = 1

def k_fold():
    # Create training and test variables 
    # K-fold is 20-80% of the total data
    k_fold_value = int(len(initial_data) / 5)
    initial_val = 0
    error_arr = []
    for j in range(k_fold_value, len(initial_data) + 1, k_fold_value):
        X_train, Y_train, X_test, Y_test = [], [], [], []
        for i, val in enumerate(initial_data):
            
            if i > initial_val and i <= j:
                test = []
                test.append(float(val[0]))
                test.append(float(val[1]))
                X_test.append(test)
                Y_test.append(float(val[2]))
            else:
                test = []
                test.append(float(val[0]))
                test.append(float(val[1]))
                X_train.append(test)
                Y_train.append(float(val[2]))
        initial_val += k_fold_value
        
        # Calculate the coefficients of the polynomial
        theta, train_arr = perceptron_train(np.array(X_train), np.array(Y_train), 0.005, 500)
        
        # Predicted value of 'y'
        y_predicted = evaluate(np.array(theta), np.array(X_test))

        # Now, comparing the value of Y_test (actual values) with the predicted values of 'y'
        error = np.sum((y_predicted - Y_test) ** 2) / len(X_train)
        error_arr.append(error)
    return error_arr

errors = k_fold()
mean_error = sum(errors) / len(errors)
print("Final mean error", mean_error)
        