# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 17:11:09 2020

@author: dDo
"""
'''
Some example for practice to understand SVM or Support Vector Machine
Without kernel (linear kernel) and with Gaussian Kernel (rbf kernel)
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.svm import SVC

'''
Find linear decision boundary using SVM in 1st example dataset withour kernel (linear kernel)
'''
#load the dataset
data = loadmat("ex6data1.mat")
X = data["X"]
y = data["y"]

#plot the data set
(m, n) = X.shape
pos, neg = (y == 1).reshape(m, 1), (y == 0).reshape(m, 1)
plt.scatter(X[pos[:, 0], 0], X[pos[:, 0], 1])
plt.scatter(X[neg[:, 0], 0], X[neg[:, 0], 1])

#as recommended, we try NOT to code SVM from scratch but instead using highly optimized
#library such as "sklearn" for this assignment
#SVC with default C = 1.0 #Then, test C = 100 in SVC(C=100, kernel="linear")
classifier = SVC(kernel="linear")
classifier.fit(X, np.ravel(y))

#plot the decision boundary
plt.figure()
plt.scatter(X[pos[:, 0], 0], X[pos[:, 0], 1])
plt.scatter(X[neg[:, 0], 0], X[neg[:, 0], 1])
X_1, X_2 = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 1].max(), num = 100),
                       np.linspace(X[:, 1].min(), X[:, 1].max(), num = 100))
plt.contour(X_1, X_2, classifier.predict(np.array([X_1.ravel(), X_2.ravel()]).T).reshape(X_1.shape), 1, colors="b")
plt.xlim(0, 4.5)
plt.ylim(1.5, 5)

'''
Use kernel in example dataset 2, which is not be linearly seperable
'''
#understand Kernel trick
#https://towardsdatascience.com/understanding-the-kernel-trick-e0bc6112ef78\

#load dataset
data2 = loadmat("ex6data2.mat")
X2 = data2["X"]
y2 = data2["y"]

#plot the dataset
(m2, n2) = X2.shape
pos2, neg2 = (y2 == 1).reshape(m2, 1), (y2 == 0).reshape(m2, 1)
plt.scatter(X2[pos2[:, 0], 0], X2[pos2[:, 0], 1], color="red")
plt.scatter(X2[neg2[:, 0], 0], X2[neg2[:, 0], 1], color="green")
plt.xlim(0, 1)
plt.ylim(0.4, 1)

#implement SVM with Gassian kernel = rbf stand for Radial Basis Function 
#in regards to parameters of SVM with rbf kernel, it uses gamma instead of sigma
#gamma nearly equal 1/sigma
classifier2 = SVC(kernel="rbf", gamma=30)
classifier2.fit(X2, np.ravel(y2))

#plot the decision boundary
plt.scatter(X2[pos2[:, 0], 0], X2[pos2[:, 0], 1], color="red")
plt.scatter(X2[neg2[:, 0], 0], X2[neg2[:, 0], 1], color="green")
X2_1, X2_2 = np.meshgrid(np.linspace(X2[:, 0].min(), X2[:, 1].max(), num = 100),
                         np.linspace(X2[:, 1].min(), X2[:, 1].max(), num = 100))
plt.contour(X2_1, X2_2, classifier2.predict(np.array([X2_1.ravel(), X2_2.ravel()]).T).reshape(X2_1.shape), 1, colors="b")
plt.xlim(0, 1)
plt.ylim(0.4, 1)

'''
Determine the best C nad gamma(sigma) values to use in example dataset 3
'''
#load dataset
data3 = loadmat("ex6data3.mat")
X3 = data3["X"]
y3 = data3["y"]
#Xval, yval are represented cross validation set
Xval = data3["Xval"]
yval = data3["yval"]

#plot dataset
(m3, n3) = X3.shape
pos3, neg3 = (y3 == 1).reshape(m3, 1), (y3 == 0).reshape(m3, 1)
#plt.figure()
plt.scatter(X3[pos3[:, 0], 0], X3[pos3[:, 0], 1], color="red")
plt.scatter(X3[neg3[:, 0], 0], X3[neg3[:, 0], 1], color="green")


#define function calculate error of validation of each pair C and gamma (model)
def dataset3Params(X, y, Xval, yval, vals):
    #return best pair of C and gamma (1/sigma) based on cross validation set
    acc = 0     #accuracy
    best_C = 0
    best_gamma = 0
    
    #which each C value corresponding to a sigma values
    #because here we use vals for both C and sigma, so it'll iterates vals^2 times
    for i in vals:
        C = i
        for j in vals:
            #gamma = 1/sigma
            gamma = 1/j
            
            #an SCV (Support Vector Classifier) model is constructed to compute 
            #the accuracy instead the error of cross-validation
            #the lower error, the higher accuracy
            classifier = SVC(C=C, gamma=gamma)
            classifier.fit(X, y)
            #score of accuracy
            score = classifier.score(Xval, yval)
            
            #print("Pair [", C, ",", j, "] \nGive:", score)
            #print("=============================")
            #store the pair of C and gamma give the highest accuracy
            if score > acc:
                acc = score
                best_C = C
                best_gamma = gamma
    
    return acc, best_C, best_gamma

#values for C and sigma
vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
acc, C, gamma = dataset3Params(X3, y3.ravel(), Xval, yval.ravel(), vals)
print("Best pair: [", C, ",", 1/gamma, "] \nGive:", acc)

#use best pair of C and gamma for training set to find decision boundary
classifier3 = SVC(C=C, gamma=gamma)
classifier3.fit(X3, np.ravel(y3))

#plot decision boundary
plt.figure()
plt.scatter(X3[pos3[:, 0], 0], X3[pos3[:, 0], 1], color="red")
plt.scatter(X3[neg3[:, 0], 0], X3[neg3[:, 0], 1], color="green")
X3_1, X3_2 = np.meshgrid(np.linspace(X3[:, 0].min(), X3[:, 1].max(), num = 100),
                         np.linspace(X3[:, 1].min(), X3[:, 1].max(), num = 100))
plt.contour(X3_1, X3_2, classifier3.predict(np.array([X3_1.ravel(), X3_2.ravel()]).T).reshape(X3_1.shape), 1, colors="b")




