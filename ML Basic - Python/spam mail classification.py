# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 21:13:56 2020

@author: dDo
"""
'''
Using SVM to build a email spam classifier
'''

#This problem is unique as it focuses more on data preprocessing than the actual modeling process.
#The emails need to process in a way that that could be used as input for the model. One way of doing 
#so is to obtain the indices of all the words in an email based on a list of commonly used vocabulary.
#Must read ex6.pdf do understand the process

import numpy as np
import pandas as pd
import re
from scipy.io import loadmat
from sklearn.svm import SVC
from nltk.stem import PorterStemmer

#load data in email sample file
file_contents = open("emailSample1.txt", "r").read()
#load vocabulary file
vocabList = open("vocab.txt", "r").read()

#store the vocabulary list as a dictionary with the vocabs as keys and indices as values
#log vocabList for easy to see
vocabList = vocabList.split("\n")[:-1]
vocabList_dic = {}
for each in vocabList:
    value, key = each.split("\t")
    vocabList_dic[key] = value
    
#Processing the email
def processEmail(email_contents, vocabList_dic):
    #Lowercase all word
    email_contents = email_contents.lower()
    
    #Handle number
    #replace all number to the text "number"
    email_contents = re.sub("[0-9]+", "number", email_contents)
    
    #Handle URLs
    #replace all URLs to the text "httpaddr"
    email_contents = re.sub("[http|https]://[^\s]*", "httpaddr", email_contents)
    
    #Handle email address
    #replace all email address to the text "emailaddr"
    email_contents = re.sub("[^\s]+@[^\s]+", "emailaddr", email_contents)
    
    #Handle $ sign
    #replace $ sign to the text "dollar"
    email_contents = re.sub("[$]+", "dollar", email_contents)
    
    #Strip all special characters
    #replace special char to nothing
    specialChar = ["<","[","^",">","+","?","!","'",".",",",":"]
    for char in specialChar:
        email_contents = email_contents.replace(str(char), "")
    email_contents = email_contents.replace("\n", " ")
    
    #Stem the word
    ps = PorterStemmer()
    #split all words to a list of words to consider each word
    email_contents = [ps.stem(token) for token in email_contents.split(" ")]
    #after stem all similar words, then join it from list to text as before
    email_contents = " ".join(email_contents)    
    
    #Process the email and return word_indices
    #compare each word from email_contents to list words in vocabList_dic
    #if it exist in vocab_dic, then replace it to an indice corresponding to that word
    word_indices = []
    for char in email_contents.split():
        if len(char) > 1 and char in vocabList_dic:
            #char here play a role as a key to get the indice value
            word_indices.append(int(vocabList_dic[char]))
    
    return word_indices

word_indices = processEmail(file_contents, vocabList_dic)

#convert word_indices into feature vector
def emailFeatures(word_indices, vocabList_dic):
    n = len(vocabList_dic)
    features = np.zeros((n, 1))
    
    for i in word_indices:
        features[i] = 1
    
    return features

#here features vector will be used to examine future email whether spam or not
features = emailFeatures(word_indices, vocabList_dic)
print("Shape of features vector:", features.shape)
print("Number of non-zero entries:", np.sum(features))

#Passing features as input to train SVM
#we need more trainning example
spam_data = loadmat("spamTrain.mat")
X_train = spam_data["X"]
y_train = spam_data["y"]

#implement SVM model with trainning set
spam_svc = SVC(C=0.1, kernel="linear")
spam_svc.fit(X_train, y_train.ravel())
acc = spam_svc.score(X_train, y_train.ravel())
print("Training accuracy:", acc)

#test model with test set
spam_data_test = loadmat("spamTest.mat")
X_test = spam_data_test["Xtest"]
y_test = spam_data_test["ytest"]

spam_svc.predict(X_test)
acc_test = spam_svc.score(X_test, y_test.ravel())
print("Test accuracy:", acc_test)    

#To better understand our model, we could look at the weights of each word and figure out 
#the words that are most predictive of a spam email.
weights = spam_svc.coef_[0]
weights_col = np.hstack((np.arange(1, 1900).reshape(1899, 1), weights.reshape(1899, 1)))

#logg to see
df = pd.DataFrame(weights_col)
df.sort_values(by=[1], ascending=False, inplace=True)

predictors = []
index = []
for i in df[0][:15]:
    for keys, values in vocabList_dic.items():
        if str(int(i)) == values:
            predictors.append(keys)
            index.append(int(values))

print("Top 15 predictors of spam:")
for _ in range(15):
    print(predictors[_],"\t\t",round(df[1][index[_]-1],6))







    