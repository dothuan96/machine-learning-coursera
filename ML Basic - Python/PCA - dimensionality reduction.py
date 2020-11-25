# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 20:51:51 2020

@author: dDo
"""
'''
PCA - Principal Component Analysis in Dimensionality Reduction
MISSING: function choosing k
'''
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd
from scipy.io import loadmat

#load data
data = loadmat("ex7data1.mat")
X = data["X"]
#visualize data
plt.scatter(X[:, 0], X[:, 1], marker="o", facecolors="none", edgecolors="b")

#feature normalization is needed to ensure that data are in the same range
def featureNormalize(X):
    #compute mean regarding column, axis=1 will compute regarding row
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    
    X_norm = (X - mu)/sigma
    return X_norm, mu, sigma

def pca(X):
    (m, n) = X.shape
    #compute covariance matrix Sigma
    sigma = 1/m * X.T @ X
    
    #use "singular value decomposition" function in numpy
    #compute eigenvector of covariance matrix
    U, S, V = svd(sigma)
    
    return U, S, V

#call the PCA function
X_norm, mu, std = featureNormalize(X)
U, S, V = pca(X_norm)
print("Top eigenvector:", U[:, 0])

#visualize data
plt.scatter(X[:, 0], X[:, 1], marker="o", facecolors="none", edgecolors="b")
plt.plot([mu[0],(mu+1.5*S[0]*U[:,0].T)[0]], [mu[1],(mu+1.5*S[0]*U[:,0].T)[1]], color="black", linewidth=3)
plt.plot([mu[0],(mu+1.5*S[1]*U[:,1].T)[0]], [mu[1],(mu+1.5*S[1]*U[:,1].T)[1]], color="black", linewidth=3)
plt.xlim(-1,7)
plt.ylim(2,8)

#to reduce the dimension of the dataset, we project the data onto the principal components (eigenvectors) found.
def projectData(X, U, K):
    m, n = X.shape
    U_reduced = U[:, :K]
    print("U_reduced shape:", U_reduced.shape)
    #z have k-dimension, so z have k column
    Z = np.zeros((m, K))
    
    #Z = X @ U_reduced
    #alternative way to compute Z
    for i in range(m):
        for k in range(K):
            Z[i, k] = X[i, :] @ U_reduced[:, k]

    return Z

#project data onto K=1 dimension
K = 1
Z = projectData(X_norm, U, K)
print("Projection of the first example:", Z[0][0])

#data also can be uncompressed back to original space
def recoverData(X, Z, U, K):
    m, n = X.shape
    X_approx = np.zeros((m, n))
    U_reduced = U[:, :K]
    
    #X_approx = Z @ U_reduced.T
    #alternative way to compute X_approximate
    for i in range(m):
        X_approx[i, :] = Z[i, :] @ U_reduced.T
    
    return X_approx

X_approx = recoverData(X, Z, U, K)
print("Appoximation of the first example:\n", X_approx[0, :])

#plot to see different between original and approximate data
plt.figure()
plt.scatter(X_norm[:,0], X_norm[:,1], marker="o", label="Original", facecolors="none", edgecolors="b", s=15)
plt.scatter(X_approx[:,0], X_approx[:,1], marker="o", label="Approximation", facecolors="none", edgecolors="r", s=15)
plt.title("The Normalized and Projected Data after PCA")
plt.legend()

'''
Work with more complex data - Face image dataset
'''
#load data
data2 = loadmat("ex7faces.mat")
X2 = data2["X"]

#visualize data
fig, ax = plt. subplots(nrows=10, ncols=10, figsize=(8,8))
for i in range(0, 100, 10):
    for j in range(10):
        ax[int(i/10), j].imshow(X2[i+j, :].reshape(32, 32, order="F"), cmap="gray")
        ax[int(i/10), j].axis("off")

#each image(tr.example) have 32x32 pixels, that's mean 1024 pixels (features) for each
X_norm2 = featureNormalize(X2)[0]

#run PCA
U2, S2, V2 = pca(X_norm2)

#visualize data after reduce the dimensional
#K=36 give us the largest variance retained
U_reduced2 = U2[:, :36].T
fig2, ax2 = plt.subplots(6, 6, figsize=(8,8))
for i in range(0, 36, 6):
    for j in range(6):
        ax2[int(i/6), j].imshow(U_reduced2[i+j, :].reshape(32, 32, order="F"), cmap="gray")
        ax2[int(i/6), j].axis("off")

#now we try projecting K=100, then recover it to understand what is lost in dimension reduction
#you also can try visualize K=100 instead of 36 on above
K2 = 100
Z2 = projectData(X_norm2, U2, K2)

#data uncompress
X_approx2 = recoverData(X2, Z2, U2, K2) 

#Visuallize uncompressed data
#what we project here are PROJECTED DATA, so it absolutely not give us image same as original
fig3, ax3 = plt.subplots(10, 10, figsize=(8,8))
for i in range(0, 100, 10):
    for j in range(10):
        ax3[int(i/10), j].imshow(X_approx2[i+j, :].reshape(32, 32, order="F"), cmap="gray")
        ax3[int(i/10), j].axis("off")















    
    