# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 19:23:55 2020

@author: dDo
"""
'''
Movie recommender system
Using Collborative Filtering algorithm and Gradient Descent 
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# load movie rating dataset
data1 = loadmat("ex8_movies.mat")
data2 = loadmat("ex8_movieParams.mat")
# 1682 X 943 matrix, containing ratings (1-5) of 1682 movies on 943 user
Y = data1["Y"]
# 1682 X 943 matrix, where R(i,j) = 1 if and only if user j give rating to movie i
R = data1["R"]
# 1682 X 10 matrix , num_movies X num_features matrix of movie features
X = data2["X"]
# 943 X 10 matrix, num_users X num_features matrix of user features
Theta = data2["Theta"]

# compute avarage rating for 1st movie
ave = np.sum(Y[0,:] * R[0,:]) / np.sum(R[0,:])
print("Average rating for movie 1 (Toy Story):", round(ave, 2), "/5")

#visualize data
plt.figure(figsize=(8, 16))
plt.imshow(Y)
plt.xlabel("Users")
plt.ylabel("Movies")

#compute cost function and gradient of Collaborative filtering
def cofiCostFunc(X, Theta, Y, R, Lambda):
    predictions = X @ Theta.T
    error = (predictions - Y)
    #element-wise multiply to R will extract the movies which have not rated
    #because r of movies have rated = 1, and have not rated = 0, so when multiply all 
    #error value corresponding to r=0 will be = 0, and error value corresponing r=1 will be kept to sum
    J = 1/2 * np.sum((error**2) * R)
    
    #compute regularized cost function
    reg_X = Lambda/2 * np.sum(X**2)
    reg_Theta = Lambda/2 * np.sum(Theta**2)
    J += reg_X + reg_Theta
    
    #compute gradient
    X_grad = error*R @ Theta + Lambda*X
    Theta_grad = (error*R).T @ X + Lambda*Theta
    
    return J, X_grad, Theta_grad

#Reduce the data set to run faster
X_test = X[:5, :3]
Theta_test = Theta[:4, :3]
Y_test = Y[:5, :4]
R_test = R[:5, :4]

#evaluating cost function
J, X_grad, Theta_grad = cofiCostFunc(X_test, Theta_test, Y_test, R_test, 0)
print("Cost at lambda = 0:", J)
J2, X_grad2, Theta_grad2 = cofiCostFunc(X_test, Theta_test, Y_test, R_test, 1.5)
print("Cost at lambda = 1.5:", J2)

#once we ensure our cost function and gradient work, then we start traing algorithm
#load movie list
movieList = open("movie_ids.txt", "r").read().split("\n")[:-1]
#print(movieList)

#initialize MY rating, 1 user
my_ratings = np.zeros((1682, 1))
#create own rating, use given rating in exercise
my_ratings[0] = 4 
my_ratings[97] = 2
my_ratings[6] = 3
my_ratings[11]= 5
my_ratings[53] = 4
my_ratings[63]= 5
my_ratings[65]= 3
my_ratings[68] = 5
my_ratings[82]= 4
my_ratings[225] = 5
my_ratings[354]= 5

#check my rating star
print("New user ratings:\n")
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print("Rated", int(my_ratings[i]), "star for index", movieList[i])
       
#before feed data to our algorithm, we need to mean normalize the ratings
def normalizeRatings(Y, R):
    m, n = Y.shape
    Ymean = np.sum(Y*R, axis=1) / np.sum(R, axis=1)
    Ymean = Ymean[:, np.newaxis]
    Ynorm = Y - Ymean
    
    return Ymean, Ynorm

Ymean, Ynorm = normalizeRatings(Y, R)
print("Y mean:", Ymean.shape)
print("Y norm:", Ynorm.shape)

#use gradient descent to training our algorithm
def gradientDescent(X, Theta, Y, R, alpha, iterations, Lambda):
    #store all J
    J_history = []
    
    for _ in range(iterations):
        #find gradient
        J, X_grad, Theta_grad = cofiCostFunc(X, Theta, Y, R, Lambda)
        #store J
        J_history.append(J)
        X = X - alpha*X_grad
        Theta = Theta - alpha*Theta_grad
    
    return J_history, X, Theta

#add new rating (my rating)
Y = np.hstack((my_ratings, Y))
R = np.hstack((my_ratings != 0, R))

#set initial parameters theta, X
#m: number movies, n: number user, k: number feature
m, n = Y.shape
k = 10
X_init = np.random.randn(m, k)
Theta_init = np.random.randn(n, k)

#optimize parameter using GD
J_history, X_optimal, Theta_optimal = gradientDescent(X_init, Theta_init, Y, R, 0.001, 400, 10)

#plot all cost function J to ensure our cost reduce each iteration, that's mean X and Theta are optimal
plt.figure()
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent") 