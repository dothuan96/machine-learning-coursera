# -*- coding: utf-8 -*-
"""
Created on Sat May 16 20:52:42 2020

@author: dDo
"""
'''
Regconize hand writing by neural network
Using Feedforward and Bacpropagation to find the gradient
MISSING: minimize cost funtion
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat

#reading data
data = loadmat('ex4data1.mat')
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

#our network have 3 layers: 
#input layer have 400 units corresponding 400 features of X for each training example (20x20 pixels)
#1 hidden layer have 25 units
#output layer have 10 units corresponding 10 labels
#the given parameters are stored in ex4weight.mat contain Theta1 and Theta2
weights = loadmat('ex4weights.mat')
#Theta1 has size 25 x 401
theta1 = weights['Theta1']
#Theta2 has size 10 x 26
theta2 = weights['Theta2']

#roll parameters together, nn_parameters has size 10285 x 1
#ndarray.ravel() return a flattened array (1D array)
nn_params = np.hstack((theta1.ravel(order='F'), theta2.ravel(order='F')))

#neural network hyperparameters
input_layer_size = 400
hidden_layer_size = 25
num_labels = 10
Lambda = 1

'''
Feedforward propagation and Cost function
'''
#original ouput were 1-10 corresponding to labels 1-10, so we need to encode y for suitable with neural network
#y should be [1 0 0 0 ...] for label 1, [0 1 0 0 0 ...] for label 2, so on
#pandas.get_dummies convert categorical var into indicator var
#y_d will has shape 5000 x 10
y_d = pd.get_dummies(y.flatten())

#define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#feedforward propagation
def unrollPara(nn_params, input_layer_size, hidden_layer_size, num_labels):
    #unroll theta1 from nn_params with shape 25 x 401
    theta1 = np.reshape(nn_params[:hidden_layer_size*(input_layer_size+1)], (hidden_layer_size, input_layer_size+1), 'F')
    #unroll theta2 from nn_params with shape 10 x 26
    theta2 = np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):], (num_labels, hidden_layer_size+1), 'F')
    
    return theta1, theta2
    
    
#define cost function
def nnCostFunc(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda):
    #it's ok if including this part or not
    theta1, theta2 = unrollPara(nn_params, input_layer_size, hidden_layer_size, num_labels)
    
    m = len(y)
    #adding intercept and assign X to a1 (matrix a for layer 1)
    ones = np.ones((m, 1))
    #a1 will has shape 5000 x 401
    a1 = np.hstack((ones, X))
    #calculate matrix a for layer 2
    a2 = sigmoid(a1 @ theta1.T)
    #adding intercept for a2 will has shape 5000 x 26
    a2 = np.hstack((ones, a2))
    #calculate matrix a for layer 3, it's also knowns as h (output) will has shape 5000 x 10
    h = sigmoid(a2 @ theta2.T)
    
    error = np.sum(np.multiply(y_d, np.log(h)) + np.multiply(1 - y_d, np.log(1 - h)))
    #cost of all output
    cost = (-1/m) * np.sum(error)
    #print('Cost:', cost)
    
    #calculate extra terms, sum of (all theta square) of (all layers), from j=1
    #sum of theta1 (theta layer 1), axis=1 mean sum row by row
    sum_theta1 = np.sum(np.sum(np.power(theta1[:, 1:], 2), axis=1))
    #sum of theta1 (theta layer 2)
    sum_theta2 = np.sum(np.sum(np.power(theta2[:, 1:], 2), axis=1))
    #extra terms
    extra_terms = (Lambda/(2*m)) * (sum_theta1 + sum_theta2)
    #print('Terms:', extra_terms)
    
    return cost + extra_terms

J = nnCostFunc(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)
print('Cost of given theta:', J)

'''
Backpropagation
'''
#define sigmoid gradient function
def sigmoidGra(z):
    return np.multiply(sigmoid(z), 1 - sigmoid(z))

#ramdon initialize the weight parameters for symmetry break
#https://stackoverflow.com/a/20029817/
def randInitializeWeights(L_in, L_out):
    epsilon = 0.12
    return np.random.rand(L_out, L_in+1) * 2 * epsilon - epsilon

#shape 25 x 401
initial_theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
#shape 10 x 26
initial_theta2 = randInitializeWeights(hidden_layer_size, num_labels)

#roll parameters into a single vector
nn_initial_params = np.hstack((initial_theta1.ravel(order='F'), initial_theta2.ravel(order='F')))

#define grdient funtion based on backprop
def nnGrad(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda):
    initial_theta1, initial_theta2 = unrollPara(nn_params, input_layer_size, hidden_layer_size, num_labels)
    #set all delta for all layer = 0
    delta1 = np.zeros(initial_theta1.shape)
    delta2 = np.zeros(initial_theta2.shape)
    m = len(y)
    
    #looping for every training set, here we have 5000 traning set
    for i in range(X.shape[0]):
        #compute feedforward to get a3 for EACH training set
        ones = np.ones((1, 1))
        #a1 will have shape 1 x 401
        a1 = np.hstack((ones, X[i][np.newaxis, :]))
        #z2 shape 1 x 25
        z2 = a1 @ initial_theta1.T
        #we use usual sigma function for computing forward prop
        a2 = sigmoid(z2)
        #a2 shape 1 x 26
        a2 = np.hstack((ones, a2))
        #z3 shape 1 x 10
        z3 = a2 @ initial_theta2.T
        a3 = sigmoid(z3)
        
        #compute error of output layer (layer 3) by getting the different between output y and a3
        #d3 is the delta (error) of layer 3
        #y dummies here still a list, so after extract y_d at i, convert it to 2D-array
        #d3 shape 1 x 10
        d3 = a3 - y_d.iloc[i, :][np.newaxis, :]
        #compute delta for hidden layer from right to left
        #we use sigma grdient function for computing backprop
        #z2 shape 1 x 26
        z2 = np.hstack((ones, z2))
        d2 = np.multiply(d3 @ initial_theta2, sigmoidGra(z2))
        
        #update delta layer
        #d2 represent for hidden layer 2, and here we have 25 units, so we remove bias term
        #delta1 shape 25 x 401
        delta1 = delta1 + d2[:, 1:].T @ a1
        #delta2 shape 10 x 26
        delta2 = delta2 + d3.T @ a2
        
    delta1 *= 1/m
    delta2 *= 1/m
    #for j from 1...m, we add extra term for its
    delta1[:, 1:] = delta1[:, 1:] + (Lambda/m) * initial_theta1[:, 1:]
    delta2[:, 1:] = delta2[:, 1:] + (Lambda/m) * initial_theta2[:, 1:]
    
    #roll into a single vector
    return np.hstack((delta1.ravel(order='F'), delta2.ravel(order='F')))

nn_backprop_params = nnGrad(nn_initial_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)
print("Shape of gradient parameter:", nn_backprop_params.shape)

'''
Gradient checking
After checking and make sure there is bug-free, we should disable gradient checking
if we don't want it make our model getting slow
'''
#define gradient checking to ensure our backprop implementation is bug free
def checkGradient(nn_initial_params, nn_backprop_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda):
    epsilon = 0.0001
    #number of elements in initial_theta_params, 10285
    n_elems = len(nn_initial_params)
    
    #pick random 10 elements
    for i in range(10):
        #get a random number from 1 - 10285
        x = int(np.random.rand()*n_elems)
        #initiallize vector for epsilon with shape = shape of initial theta parameter
        esp_vec = np.zeros(n_elems)
        #set the value at x = epsilon
        esp_vec[x] = epsilon
        
        #adding & subtracting epsilon for the value at the position x in initial theta
        add_eps = nn_initial_params + esp_vec
        sub_eps = nn_initial_params - esp_vec
        #calculate the cost
        cost_high = nnCostFunc(add_eps, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)
        cost_low  = nnCostFunc(sub_eps, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)
        #compute numerical gradient
        myGrad = (cost_high - cost_low) / (2 * epsilon)
        
        #compare to backprop parameter at position x
        print("Element: {0}. \nNumerical Gradient = {1:.9f}. \nBackProp Gradient = {2:.9f}. \n".format(x, myGrad, nn_backprop_params[x]))
        
#checkGradient(nn_initial_params, nn_backprop_params, input_layer_size, hidden_layer_size, num_labels, X, y, 0.)

'''
Optimize the parameter
When we have gradient, we can use gradient descend or a built-in optimization function (such as fmincg)
to minimize the cost function with weights in theta
'''


