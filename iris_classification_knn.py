# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 12:52:46 2020

@author: dDo
"""
# Iris Classification using KNN: The iris flowers have different species and 
# you can distinguish them based on the length of petals and sepals.

# import iris dataset form sklearn library
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

# iris is a dictionary contain 150 examples:
# "data" key: 4 features of iris: sepal length, sepal width, petal length, petal width
# "target" key: label for 3 spieces of iris 0, 1, 2
# "target_names" key: name of 3 spieces of iris 'setona', 'versicolor', 'virginica'
# etc.
iris = load_iris()
# "DESCR" key: Description/ References for dataset
# print(iris["DESCR"])

'''
# split 4 features into 4 vectors
features = iris.data.T    # (4, 150)
sepal_length = features[0]
sepal_width = features[1]
petal_length = features[2]
petal_width = features[3]

# get label for each feature
sepal_length_label = iris.feature_names[0]
sepal_width_label = iris.feature_names[1]
petal_length_label = iris.feature_names[2]
petal_width_label = iris.feature_names[3]

# visuallize
plt.scatter(sepal_length, sepal_width, c=iris.target)
plt.xlabel(sepal_length_label)
plt.ylabel(sepal_width_label)
plt.show()
'''

# split data into training set and testing set with corresponding y set
# train_test_split split arrays or matrices into RANDOM train and test subsets
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)

# implement KNN/ k-nearest neighbors vote
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# create a sample data to test model, also have 4 features as training data
X_new = np.array([[5.0, 2.9, 1.0, 0.2]])
prediction = knn.predict(X_new)
print(prediction)

# evaluate how accurate model is
print('Accuracy:', knn.score(X_test, y_test))