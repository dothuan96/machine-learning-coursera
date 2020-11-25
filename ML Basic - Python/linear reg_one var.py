#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 13:08:14 2020

@author: thuando
"""
"""
Linear regression for 1 variable
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading dataset
data = pd.read_csv('ex1data1.txt', header = None)
#read 1st column
X = data.iloc[:,0]
#read 2nd column
y = data.iloc[:,1]
#number of training examples
m = len(y)
#view first few rows of the data
data.head()

plt.scatter(X,y)
plt.xlabel('Population of city (10,000s)')
plt.ylabel('Profit in (10,000s)')
plt.show()

#convert rank-1 array to rank-2 array
X = X[:, np.newaxis]
y = y[:, np.newaxis]
#initialize the initial parameter theta_0 and theta_1 = 0
theta = np.zeros([2,1])
iterations = 2000
#learning rate
alpha = 0.01
#create intercept x0 = 1
ones = np.ones((m,1))
#adding intercept
X = np.hstack((ones, X))

#computing the cost
def computeCost(X, y, theta):
    #multiply matrix X and vector theta = h_theta(x)
    error = np.dot(X, theta) - y
    return np.sum(np.power(error, 2)) / (2*m)

J = computeCost(X, y, theta)
print("Cost of random theta:", round(J, 2))

#finding optimal parameters using Gradient Descent
def gradientDescent(X, y, theta, alpha, iterations):
    #update 'interations' times
    for _ in range(iterations):
        #find hypothesis - y
        error = np.dot(X, theta) - y
        #X.T mean transpose X
        #error * x_(i) 
        gradient = np.dot(X.T, error)
        theta = theta - (alpha/m) * gradient
    return theta

thetaOptimal = gradientDescent(X, y, theta, alpha, iterations)
print("Theta optimal:", np.round(thetaOptimal, 2))
print("Cost of theta optimal:", round(computeCost(X, y, thetaOptimal), 2))

#plot showing the best fit line
hyps = np.dot(X, thetaOptimal)
plt.scatter(X[:,1], y)
plt.xlabel('Population of city (10,000s)')
plt.ylabel('Profit in (10,000s)')
plt.plot(X[:,1], hyps)
plt.show()
