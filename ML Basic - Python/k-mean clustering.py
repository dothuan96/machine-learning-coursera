# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:32:39 2020

@author: dDo
"""
'''
K-mean clustering algorithm
MISSING: optimize the initial centroids by using cost (distortion) function
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.io import loadmat

#load data
data = loadmat("ex7data2.mat")
X = data["X"]
#unsupervisor doesn't have label so there is no y

#since K-means algorithms do not always give the optimal solution, random initialization is important.
def kMeanInitCentroids(X, K):
    (m, n) = X.shape
    centroids = np.zeros((K, n))
    
    for i in range(K):
        #random K tr.examples in X, then take it to initial centroids
        centroids[i] = X[np.random.randint(0, m+1), :]
    
    return centroids

#find the closest centroids by evaluating distance between training example and each centroid
#Cluster assignment step
def findClosestCentroids(X, centroids):
    (m, n) = X.shape
    #number of centroids
    K = centroids.shape[0]
    #index of centroid we will assign to each training example c(i)
    idx = np.zeros((m, 1))
    temp = np.zeros((K, 1))
    
    #assign each training eaxample
    for i in range(m):
        for j in range(K):
            #calculate distance between training example i and each K
            distance = np.sum((X[i, :] - centroids[j, :])**2)
            #store the the distance of i to ALL K centroids into temp
            temp[j] = distance
        # argmin find the smallest distance(value) in temp and return indices(position) of it in array, and assign it to training example i
        #because indice in python start form 0, so +1 used here to number the centroid from 1 instead 0
        idx[i] = np.argmin(temp) + 1
        
    return idx

#sinitialize centroid
K = 3
initial_centroids = kMeanInitCentroids(X, K)
print("Initial centroids:\n", initial_centroids)
idx = findClosestCentroids(X, initial_centroids)
print("Closest centroids for the first 3 examples:\n", idx[0:3])

#compute mean of all training examples which assigned to K centroids
#Move centroid step
def computeCentroids(X, idx, K):
    (m, n) = X.shape
    #store sum of training examples have been assigned to K, i.e. centroids[1] store sum of training examples assiged to group 1
    centroids = np.zeros((K, n))
    #store number of training examples have been assigned to K
    count = np.zeros((K, 1))
    #for easy to work, we plain the idx matrix (which contain array for each index), and python work from 0, so we must -1
    idx_plain = idx.ravel() - 1
    
    #instead of choose which training examples assigned to which K in a messy order, we will go through all tr.examples and sum them into the centroids[indices], indices of centroids based on idx
    for i in range(m):
        #get the index (centroids) of tr.eaxmple i 
        index = int(idx_plain[i])
        #now we know which centroid (index) have been assigned to tr.example i, so we sum its tr.example to centroids[index]
        centroids[index, :] += X[i, :]
        count[index] += 1
    
    return centroids/count

centroids = computeCentroids(X, idx, K)
print("Centroids after we move based on the mean of all training examples belonged to K centroids:\n", centroids)

#plot to many different subplot to see the movement of the centroids each time
def plotKmeans(X, centroids, idx, K, num_iters):
    (m, n) = X.shape
    fig, ax = plt.subplots(nrows=num_iters, ncols=1, figsize=(6,36))
    
    for i in range(num_iters):
        #visualize data
        color = "rgb"
        for k in range(1, K+1):
            group = (idx == k).reshape(m, 1)
            ax[i].scatter(X[group[:, 0], 0], X[group[:, 0], 1], c=color[k-1], s=15)

        #visualize the new centroids
        ax[i].scatter(centroids[:, 0], centroids[:, 1], s=120, marker="x", c="black", linewidth=3)
        title = "Iteration Number " + str(i)
        ax[i].set_title(title)
        
        #move the centroids
        centroids = computeCentroids(X, idx, K)
        #assign each tr.example to closest new centroids
        idx = findClosestCentroids(X, centroids)
        
    plt.tight_layout()

#call the function    
plotKmeans(X, centroids, idx, K, 6)

'''
Now we can implementing above algorithm for other dataset.
Compress image to 16 color base (16 clusters)
'''
#load data
data2 = loadmat("bird_small.mat")
A = data2["A"]
#log A and A.shape to see, A is 3D array

#prepocess and reshape the data of image to 2D array
#read more: ex7.pdf
X2 = (A/255).reshape(128*128, 3)

#implement k-mean algorithm
def runKmeans(X, initial_centroids, num_iters, K):
    idx = findClosestCentroids(X, initial_centroids)
    
    for i in range(num_iters):
        #move the centroids
        centroids = computeCentroids(X, idx, K)
        #re-assign tr.examples to new centroids
        idx = findClosestCentroids(X, centroids)
    
    return idx, centroids

#run k-mean algorithm on the new dataset
K2 = 16
num_iters = 10
initial_centroids2 = kMeanInitCentroids(X2, K2)
#optimal centroids after n interations
idx2, centroids2 = runKmeans(X2, initial_centroids2, num_iters, K2)

(m2, n2) = X2.shape
X2_compress = X2.copy()
#replace each pixel to the similar color in 16 given colors
#idx2==k will return a matrix True/ False, which tr.example in X2_compress coressponding to True value in its matrix, then replace it
#for k in range(1, K2+1):
    #X2_compress[(idx2 == k).ravel(), :] = centroids2[k-1]

#alternative way to replace similar color
idx2_plain = idx2.ravel() - 1
for i in range(m2):
    #find tr.example i belonged to which cluster (k)
    index = int(idx2_plain[i])
    X2_compress[i] = centroids2[index]
    
#reshape the compressed data to 3D array
X2_compressed = X2_compress.reshape(128, 128, 3)

#display the image
fig, ax = plt.subplots(1, 2)
ax[0].imshow(A)
ax[1].imshow(X2_compressed) 