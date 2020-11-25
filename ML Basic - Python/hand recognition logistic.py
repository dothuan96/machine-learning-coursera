# -*- coding: utf-8 -*-
"""
Created on Mon May 11 21:03:43 2020

@author: dDo
"""
'''
Regconize hand writing by One-vs-all classification method
The given data include 5000 training set with 400 features (20x20 pixel)
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.io import loadmat

#using loadmat to load data from matlab file
data = loadmat('ex3data1.mat')
X = data['X']
y = data['y']

#visuallize data
# 0 is labeled as 10 while 1-9 are labeled as 1-9
fig, axes = plt.subplots(10, 10, figsize=(8, 8))
for i in range(10):
    for j in range(10):
        axes[i, j].imshow(X[np.random.randint(0, 5001), :]\
                          .reshape(20, 20, order="F"), cmap="hot")
        axes[i, j].axis("off")
        
#define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#define Regularization cost function, here we have extra term controlled by lamda
def costFunctionReg(X, y, theta, Lambda):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    error = np.multiply(y, np.log(h)) + np.multiply((1 - y), np.log(1 - h))
    #calculate regularization term
    reg = (Lambda/(2*m)) * np.sum(theta[1:]**2)
    cost = (-1/m) * np.sum(error)
    return cost + reg

#define gradient
def gradient(X, y, theta, Lambda):
    m = len(y)
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

#use some test case to test the function
theta_t = np.array([-2, -1, 1, 2]).reshape(4, 1)
X_t = np.array([np.linspace(0.1, 1.5, 15)]).reshape(3, 5).T
X_t = np.hstack((np.ones((5, 1)), X_t))
y_t = np.array([1, 0, 1, 0, 1]).reshape(5, 1)
J = costFunctionReg(X_t, y_t, theta_t, 3)
print("Cost:", J, "\nExpected cost: 2.534819")
grad = gradient(X_t, y_t, theta_t, 3)
print("Gradients:\n", grad, "\nExpected gradients:\n 0.146561\n -0.548558\n 0.724722\n 1.398003")

#finding the optimal parameters using GD step by step
def gradientDescentMulti(X, y, theta, Lambda, alpha, iterations):
    #update 'interations' times
    for _ in range(iterations):
        #gradient is already + reg in gradient function
        grad = gradient(X, y, theta, Lambda)
        theta = theta - alpha * grad
    return theta

#define initial parameter
(m, n) = X.shape
theta = np.zeros(((n+1), 1))
#add intercept terms then X will have n+1 col
X = np.hstack((np.ones((m, 1)), X))

#since we have more than one class, we will have to train multiple 
#logistic regression classifiers using one-vs-all classification method 
def oneVsAll(X, y, theta, num_labels, Lambda):
    #Takes in numpy array of X,y, int num_labels and float lambda to train multiple logistic regression classifiers
    #depending on the number of num_labels using gradient descent. 

    #create an amty ndarray with proper dimension with theta
    all_theta = np.empty((0, n+1))
    
    #Returns a matrix of theta, where the i-th row corresponds to the classifier for label i
    for i in range(1, num_labels+1):
        #np.where return an array that all values sastify the condition will be 1, and the rest will be 0
        thetaOptimal = gradientDescentMulti(X, np.where(y==i, 1, 0), theta, Lambda, 1, 300)
        #adding each label corresponding each row
        all_theta = np.vstack((all_theta, thetaOptimal.T))
    return all_theta
        
all_theta = oneVsAll(X, y, theta, 10, 0.1)
print("10 row theta corresponding 10 labels\nAll_theta shape:", all_theta.shape)

#check the accuracy of the model
#calculate hypothesis
h = np.dot(X, all_theta.T)
#https://stackoverflow.com/questions/36300334/understanding-argmax
prediction = np.argmax(h, axis=1) + 1
#compare with output y
check = (prediction.reshape(m, 1) == y)
acc = np.mean(check) * 100 
print('Train accuracy:', acc)
    
    