#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 13:52:27 2020

@author: thuando
"""
'''
Logistic regression
Cost  function & gradient descent
MISSING: Plot decision boundary
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read data
data = pd.read_csv('ex2data1.txt', header = None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
data.head()

#visualize the data
#plot values represent for students admitted
pos , neg = (y == 1).reshape(100,1) , (y == 0).reshape(100,1)
plt.scatter(X[pos[:,0],0],X[pos[:,0],1],c="r",marker="+")
#plot values represent for students not admitted 
plt.scatter(X[neg[:,0],0],X[neg[:,0],1],marker="o",s=10)
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend(["Admitted","Not admitted"],loc=0)

#define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#implement the cost function for Logistic Regression
def costFunction(theta, X, y):
    #np.multiply does an "element-wise multiplication"
    #np.dot is the "dot product"
    h = sigmoid(np.dot(X, theta))
    error = np.multiply(y, np.log(h)) + np.multiply((1 - y), np.log(1 - h))
    return (-1/m)*np.sum(error)

# NOTE: Before doing GD, NEVER forget to do feature scaling 
#for a multivariate problem
# axis=0 mean calculate mean and standard deviation for EACH column
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

#get number of rows and column
# m: number of rows (training examples)
# n: number of columns (fetures)
(m, n) = X.shape
#create and adding intercept
ones = np.ones((m, 1))
X = np.hstack((ones, X))
#convert rank-1 to rank-2 array (2D array)
y = y[:, np.newaxis]
#initialize theta with all 0
#theta with n+1 rows (include x_0 we add to X, that's mean X have n+1 features)
theta = np.zeros((n+1, 1))

J = costFunction(theta, X, y)
print('Cost of random theta (0):', round(J, 3))

#define gradient
def gradient(X, y, theta):
    # NOTE: even its look identical to linear regression gradient
    # hypothesis in logistic regression is sigmoid function
    h = sigmoid(np.dot(X, theta))
    error = h - y
    #X.T mean transpose X
    #error * x_(i) 
    gradient = (1/m) * np.dot(X.T, error)
    return gradient

#finding the optimal parameters using GD step by step
def gradientDescentMulti(X, theta, alpha, iterations):  
    #update 'interations' times
    for _ in range(iterations):
        grad = gradient(X, y, theta)
        theta = theta - alpha * grad
    return theta
    
thetaOptimal = gradientDescentMulti(X, theta, 0.01, 1000)
print("Theta optimal:\n", np.round(thetaOptimal, 2))
J_Optimal = costFunction(thetaOptimal, X, y)
print('Cost of theta optimal:', round(J_Optimal, 3))

#plotting decision boundary
#because we plot "Exam 1 score" (feature x_1) for x axis and "Exam 2 score" (feature x_2) for y axis
#check the accuracy of the model