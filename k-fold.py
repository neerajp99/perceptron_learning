import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import random

# Import data and create a 2-D matrix out of it
data = open('hw2data-1.txt', 'r') 
Lines = data.readlines()
initial_data = []
for i, line in enumerate(Lines):
    initial_data.append(Lines[i].strip().split(' '))

def k_fold(degree):
    # Create training and test variables 
    # K-fold is 20-80% of the total data
    k_fold_value = int(len(initial_data) / 5)
    initial_val = 0
    error_arr = []
    for j in range(k_fold_value, len(initial_data), k_fold_value):
        X_train, Y_train, X_test, Y_test = [], [], [], []
        for i, val in enumerate(initial_data):
            if i > initial_val and i <= j:
                X_test.append(float(val[0]))
                Y_test.append(float(val[1]))
            else:
                X_train.append(float(val[0]))
                Y_train.append(float(val[1]))
        initial_val += k_fold_value
        # Calculate the coefficients of the polynomial
        theta = np.polyfit(np.array(X_train), np.array(Y_train), degree)

        # Predicted value of 'y'
        y_predicted = np.polyval(theta, np.array(X_test))
        
        # Now, comparing the value of Y_test (actual values) with the predicted values of 'y'
        # Calcuating the mean squared error
        error = np.sum((y_predicted - Y_test) ** 2) / len(X_test)
        error_arr.append(error)
    return error_arr

errors = k_fold(9)
mean_error = sum(errors) / len(errors)
print('Final mean error', mean_error)