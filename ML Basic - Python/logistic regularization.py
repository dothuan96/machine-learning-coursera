# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:20:08 2020

@author: dDo
"""
'''
Regularization for logistic regression
MISSING: plot decition boundary
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read data
data = pd.read_csv('ex2data2.txt', header = None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
data.head()

#visualize data
m = len(y)
pos, neg = (y == 1).reshape(m, 1), (y == 0).reshape(m, 1)
#plot feature_1 for x axis & feature_2 for y axis
#plot true vales (y == 1)
passed = plt.scatter(X[pos[:,0], 0], X[pos[:,0], 1])
#plot false values (y == 0)
failed = plt.scatter(X[neg[:,0], 0], X[neg[:,0], 1])
plt.xlabel("Test 1")
plt.ylabel("Test 2")
plt.legend((passed, failed), ('Passed', 'Failed'))
plt.show()

#add polymonial terms up to 6th power caused normal logistic regression only can fit linear desision boundary
#we'll create more data point based on x1 & x2, degree = 6 (6th power)
def mapFeature(X1, X2, degree):
    #take in numpy array of X1, X2 and return all polynomial terms up to given degree
    out = np.ones(m).reshape(m, 1)
    for i in range(1, degree+1):
        for j in range(i + 1):
            terms = (X1**(i-j) * X2**j).reshape(m, 1)
            out = np.hstack((out, terms))
    return out
#rewrite X with new fetures, X0 = 1 is already when declare out in mapFeature func
X = mapFeature(X[:, 0], X[:, 1], 6)

#define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#define Regularization cost function, here we have extra term controlled by lamda
def costFunctionReg(X, y, theta, Lambda):
    h = sigmoid(np.dot(X, theta))
    error = np.multiply(y, np.log(h)) + np.multiply((1 - y), np.log(1 - h))
    #calculate regularization term
    reg = (Lambda/(2*m)) * np.sum(theta[1:]**2)
    cost = (-1/m)*np.sum(error)
    return cost + reg

#define gradient
def gradient(X, y, theta, Lambda):
    # NOTE: even its look identical to linear regression gradient
    # hypothesis in logistic regression is sigmoid function
    h = sigmoid(np.dot(X, theta))
    error = h - y
    gradient = (1/m) * np.dot(X.T, error)
    #in gradient of regularization, j start from 1...n
    reg = (Lambda/m) * theta[1:]
    #ignore the 0th value, rewrite value from 1st...nth
    gradient[1:] = gradient[1:] + reg
    return gradient

#define initial parameter
(m, n) = X.shape
#reshape y to vector m row and 1 col
y = y.reshape(m, 1)
theta = np.zeros((n, 1))
Lambda = 1

J = costFunctionReg(X, y, theta, Lambda)
print('Cost of initial theta (0):', J)

#finding the optimal parameters using GD step by step
def gradientDescentMulti(X, y, theta, Lambda, alpha, iterations):
    #update 'interations' times
    for _ in range(iterations):
        #gradient is already + reg in gradient function
        grad = gradient(X, y, theta, Lambda)
        theta = theta - alpha * grad
    return theta

thetaOptimal = gradientDescentMulti(X, y, theta, 0.2, 1, 800)
print("Theta optimal:\n", thetaOptimal)
J_Optimal = costFunctionReg(X, y, thetaOptimal, Lambda)
print('Cost of theta optimal:', J_Optimal)

#check the accuracy of the model
#calculate hypothesis and return values to true or false in order to classify
h = np.dot(X, thetaOptimal) > 0
check = (h == y)
acc = np.mean(check) * 100 
print('Train accuracy:', acc)