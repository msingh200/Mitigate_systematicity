##importing the preprocessing function from data_processing_individual_models.py file

from data_preprocessing_individual_models.py import preprocessingPipeline

#Importing all the necessary libraries

##importing all the libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import random
import pickle
import shutil
import os
import sys
from statistics import mean


##From data_preprocessing_individual_models.py, imported the preprocessing function and preprocessing the data
X, y=preprocessingPipeline('data.csv')

##From data_preprocessing_individual_models.py, we saw that the best model for this data was Decision Tree Classifier
##My approach to mitigate systemacity is add a random normal error with mean 0 and standard deviation 0.25 (the standard deviation can be changed according to how much randomness we want to introduce to mitigate systemacity)
###Suppose the decision tree predicts 0 (no-hire) for an individual, we still want to randomly give 6.25% individuals assigned a 0 by decision tree, a prediction of 1. 
###It does not have to be 6.25%, it can be any % say 2.25% or 13% or 25%. We can adjust the predict function according to the randomness we want to generate to mitigate systemacity.

###Doing a train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

##Weights to balance out the classes 
weights= {0:0.72, 1:1.64}

clf=tree.DecisionTreeClassifier(max_depth=10, class_weight=weights)
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

## ***********************************PREDICT FUNCTION***********************************************************
##Time NOW for the PREDICT function to mitigate systemacity

def predict():  
    F_pred=[]
    for i in y_pred:
        err_pred=i+ np.random.normal(0, 0.25)
        if ((i==1) and (err_pred<0.625)):
            final_pred=0
        elif ((i==1) and (err_pred>=0.625)):
            final_pred=1
        elif ((i==0) and (err_pred>0.375)):
            final_pred=1
        elif ((i==0) and (err_pred<=0.375)):
            final_pred=0
        else:    
        final_pred=y_pred    
        F_pred.append(final_pred)
    
    return F_pred


'''Now we generate predictions from running the test data 10,000 through the predict function. Remember each time will be a bit different due to the random normal error we have added to the decision tree prediction.
Without the random normal error, all 10,000 prediction vectors from the decision tree would be the same.
Running the test data 10,000 times through the predict function, We will see the average correlation between the 10,000 error vectors generated from running the test data through predict 
function 10,000 times
We also calculate the accuracy, precision and recall.'''

##Running the test data through the predict function 10,000 times. Each time the predict function on the same test data generates slightly different predictions due to the random normal term added.


Random_added_y_pred=[]
Random_added_error_pred=[]
for _ in range(10000):
    p=predict()
    Err=y_test-p
    p=np.array(p)
    Err=np.array(Err)
    Random_added_y_pred.append(p)
    Random_added_error_pred.append(Err)


##Calculating the average accuracy, precision and recall over the 10,000 prediction vectors##

print('Accuracy = ', format(f1_score(y_test, Random_added_y_pred[i], average="micro")))

recall=[]
for i in Random_added_y_pred[i]:
    recall.append(recall_score(y_test, Random_added_y_pred[i]))

precision=[]
for i in Random_added_y_pred[i]:
    precision.append(precision_score(y_test, Random_added_y_pred[i]))
    
print('Recall = ', format(round(mean(recall), 2)))
print('Precision = ', format(round(mean(precision), 2)))

###Now we will calculate the average correlation between errors running the test data through predict function 10000 times.

##First, calculating correlation coefficient between the different prediction vectors. Corr Matrix size 10,000*10,000
corr=np.corrcoef(Random_added_error_pred, rowvar=True)


##We want the average so defining functions to calculate the average
##Upper_sum function gives the sum of correlation coefficients of the upper triangular matrix (excluding diagonal elements)
##count_upper_triangular function gives the number of elements in the upper triangular matrix (excluding the diagonal elements)
##upper_sum_divided_by_cells gives the average correlation coefficient between the 10,000 prediction vectors.

def count_upper_triangular(n):
    return ((n*n)-n)/2

def Upper_Sum(mat, r, c):
 
    i, j = 0, 0
    upper_sum = 0
 
    # To calculate sum of upper triangle
    for i in range(r):
        for j in range(c):
            if (i < j):
                upper_sum += (mat[i][j])
 
    return upper_sum

def upper_sum_divided_by_cells(mat, r, c, n):
    return Upper_Sum(mat, r, c)/count_upper_triangular(n)

##Now that we have the functions defined, we can just call it on the corr matrix of the 10,000 prediction vectors.
##Printing the average correlation between the 10000 predictions on the same test data

print('Average error correlation :', upper_sum_divided_by_cells(corr,10000,10000,10000))





