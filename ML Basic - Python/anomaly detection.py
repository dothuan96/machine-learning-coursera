# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 13:50:03 2020

@author: dDo
"""
'''
Anomaly Detection using Multivariate Gaussian
MISSING: Normal Gaussian distribution
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

#load data
data = loadmat("ex8data1.mat")
X = data["X"]
Xval = data["Xval"]
yval = data["yval"]

#visualizing data
plt.scatter(X[:, 0], X[:, 1], marker="x")
plt.xlim(0, 30)
plt.ylim(0, 30)
plt.xlabel("Latency (ms)")
plt.ylabel("Throughput (mb/s")

#fit parameter (mean and variance)
def estimateGaussian(X):
    m, n = X.shape
    #compute mean
    sum_ = np.sum(X, axis=0)
    mu = (1/m) * sum_
    #compute variance
    var = (1/m) * np.sum((X - mu)**2, axis=0)
    
    return mu, var

mu, var = estimateGaussian(X)

#define multivariate Gaussian to compute probability density
def multivariateGaussian(X, mu, var):
    m, n = X.shape
    #for work with multivariate, we use covariance matrix sigma instead of variance
    #convert variance to diagnose matrix
    sigma = np.diag(var)
    
    p = 1/((2*np.pi)**(n/2) * (np.linalg.det(sigma)**(1/2))) * np.exp((-1/2) * np.sum((X-mu)**2 @ np.linalg.pinv(sigma), axis=1))
    return p

p = multivariateGaussian(X, mu, var)
print(p.shape)

#visualize the fit
#see how to plot contour: https://www.python-course.eu/matplotlib_contour_plot.php
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], marker="x")
X1,X2 = np.meshgrid(np.linspace(0, 35, num=70), np.linspace(0, 35, num=70))
p2 = multivariateGaussian(np.hstack((X1.flatten()[:, np.newaxis], X2.flatten()[:, np.newaxis])), mu, var)
contour_level = 10**np.array([np.arange(-20 , 0, 3, dtype=np.float)]).T
plt.contour(X1, X2, p2[:, np.newaxis].reshape(X1.shape), contour_level.ravel())
plt.xlim(0, 35)
plt.ylim(0, 35)
plt.xlabel("Latency (ms)")
plt.ylabel("Throughput (mb/s)")

#select threshold epsilon and flag which examples as anomalies
def selectThreshold(yval, pval):
    #find the best threshold based on F1-score of each example in cross-validation set
    best_eps = 0
    best_F1 = 0
    
    #the range of probability (pval) we have found with training set
    step_size = (max(pval) - min(pval)) / 1000
    #create an array from pval.min to pval.max, next element > previous element step_size
    eps_range = np.arange(pval.min(), pval.max(), step_size)
    
    #loop through each epsilon
    for eps in eps_range:
        #check every single pval
        predictions = (pval < eps)[:, np.newaxis]
        #see more ex8.pdf
        #true positive
        tp = np.sum(predictions[yval == 1] == 1)
        #false positive
        fp = np.sum(predictions[yval == 0] == 1)
        #false negative
        fn = np.sum(predictions[yval == 1] == 0)
        
        #compute precision, recall and F1-score
        prec = tp / (tp + fp)
        rec  = tp / (tp + fn)
        F1 = (2*prec*rec) / (prec+rec)
        
        #choose the epsilon give us highest F1-score
        if F1 > best_F1:
            best_F1 = F1
            best_eps = eps
            
    return best_eps, best_F1

#fit parameter to X cv set
pval = multivariateGaussian(Xval, mu, var)
epsilon, F1 = selectThreshold(yval, pval)
print("Best epsilon found by using cv set:", epsilon)
print("Best F1 score on cv set:", F1)

#Visualizing the optimal threshold
plt.figure(figsize=(8, 6))

# plot the data
plt.scatter(X[:, 0], X[:, 1], marker="x")

# potting of contour
X1, X2 = np.meshgrid(np.linspace(0, 35, num=70), np.linspace(0, 35,num=70))
p2 = multivariateGaussian(np.hstack((X1.flatten()[:, np.newaxis], X2.flatten()[:, np.newaxis])), mu, var)
contour_level = 10**np.array([np.arange(-20, 0, 3, dtype=np.float)]).T
plt.contour(X1, X2, p2[:,np.newaxis].reshape(X1.shape), contour_level.ravel())

# Circling of anomalies
outliers = np.nonzero(p < epsilon)[0]
plt.scatter(X[outliers, 0], X[outliers, 1], marker ="o", facecolor="none", edgecolor="r", s=70)
plt.xlim(0, 35)
plt.ylim(0, 35)
plt.xlabel("Latency (ms)")
plt.ylabel("Throughput (mb/s)")

'''
As for high dimensional dataset, we just have to follow the exact same steps as before
'''
data2 = loadmat("ex8data2.mat")
X2 = data2["X"]
Xval2 = data2["Xval"]
yval2 = data2["yval"]

#compute mean & variance
mu2, var2 = estimateGaussian(X2)

#probability density of training set
p3 = multivariateGaussian(X2, mu2, var2)

#probability density of cross-validation set
pval2 = multivariateGaussian(Xval2, mu2, var2)

#find the best threshold
epsilon2, F1_2 = selectThreshold(yval2, pval2)
print("Best epsilon found by using cv set 2:", epsilon2)
print("Best F1 score on cv set 2:", F1_2)
print("# Outliers found:", np.sum(p3 < epsilon2))



        





