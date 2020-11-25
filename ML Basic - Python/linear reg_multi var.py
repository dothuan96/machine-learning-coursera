#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 19:21:25 2020

@author: thuando
"""
"""
Linear regression for multiple variables
"""
import numpy as np
import pandas as pd

#reading dataset
data = pd.read_csv('ex1data2.txt', sep = ',', header = None)
#read 2 first column
X = data.iloc[:, 0:2]
#read third column
y = data.iloc[:, 2]
#number of training examples
m = len(y)
#view a few first row of data
data.head()

#when features differ by orders of magnitude, we need to scale them to same range
#performing Stadardization (or Z-score Nomalization) to make GD converge much more quickly
#subtract the mean value of each feature, then divide their Standard Deviation
X = (X - np.mean(X)) / np.std(X)

#convert rank-1 array to rank-2 array
y = y[:, np.newaxis]
#initialize the initial parameter theta = 0, here we have 3 features
theta = np.zeros((3,1))
iterations = 2000
#learning rate
alpha = 0.01
#create intercept x0 = 1
ones = np.ones((m,1))
#adding intercept
X = np.hstack((ones, X))

#compute the cost
def computeCostMulti(X, y, theta):
    error = np.dot(X, theta) - y
    return np.sum(np.power(error, 2)) / (2*m)

J = computeCostMulti(X, y, theta)
print("Cost of random theta:", round(J, 2))

#finding the optimal parameters using GD
def gradientDescentMulti(X, y, theta, alpha, iterations):
    #update 'interations' times
    for _ in range(iterations):
        #find error = hypothesis - y
        error = np.dot(X, theta) - y
        #X.T mean transpose X
        #error * x_(i) 
        gradient = np.dot(X.T, error)
        theta = theta - (alpha/m) * gradient
    return theta

thetaOptimal = gradientDescentMulti(X, y, theta, alpha, iterations)
print("Theta optimal:", np.round(thetaOptimal, 2))
print("Cost of theta optimal:", round(computeCostMulti(X, y, thetaOptimal), 2))
