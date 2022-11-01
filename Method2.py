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

##Running the pre-processing function on the data, doing the train-test split, and making an array of models to chose from at random for the predict() function

X, y=preprocessingPipeline('data.csv')
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
weights= {0:0.72, 1:1.64}

MODELS=[('SVM',SVC(gamma=0.1, C=1000, class_weight=weights)) ,\
        ('SVM_linear',LinearSVC(tol=1e-5, class_weight=weights)),\
        ('Decision_tree', tree.DecisionTreeClassifier(max_depth=10, class_weight=weights)),\
        ('Random_forest', RandomForestClassifier(n_estimators=30, min_samples_split=180, class_weight=weights)),\
       ('KNN', KNeighborsClassifier(n_neighbors=10)),\
       ('Naive_Bayes', ComplementNB()),\
       ('Logistic Regression', LogisticRegression(C=10, class_weight=weights)),\
       ('AdaBoostClassifier', AdaBoostClassifier(n_estimators=100)),\
       ('MLPClassifier', MLPClassifier(solver='lbfgs', hidden_layer_sizes=(16, 6), random_state=1)),\
       ('QDA', QuadraticDiscriminantAnalysis())]

##****************************************************PREDICT FUNCTION******************************************************************************
###Finally it is time for the PREDICT function
def predict(X_test):
    randomModels = random.sample(MODELS, 3)
    y_pred = []
        
#Voting
    
    voting = VotingClassifier(estimators=randomModels, voting ='hard')
    voting.fit(x_train, y_train)
    y_pred = voting.predict(X_test)
    return y_pred

###Now it is time to test whether the predict function works. We want to see the accuracy, precision, recall and correlation of errors running the predict function through the test data ideally 10,000 times. 
###But given high computatuon time for this, I run the test data through the predict function 20 times and see the metrics. Given more time I would run it 10,000 times like I did in case1. 
###However running 20 times should still give us a good idea.


df_pred = pd.DataFrame()
err = pd.DataFrame()
finalPred = []
errorPred=[]
for i in range(20):
    finalPred.append(predict(x_test))
    errorPred.append(y_test-predict(x_test))

df_pred['Final Pred # {}'.format(i+1)] = finalPred
err['err # {}'.format(i+1)] = errorPred

##Time to get the accuracy, precision and recall values
print('Accuracy = ', format(f1_score(y_test, finalPred[i], average='micro')))

##Precision
precision=[]
for i in Random_added_y_pred[i]:
    precision.append(precision_score(y_test, finalPred[i]))

##Recall

recall=[]
for i in Random_added_y_pred[i]:
    recall.append(recall_score(y_test, finalPred[i]))


print('Recall = ', format(round(mean(recall), 2)))
print('Precision = ', format(round(mean(precision), 2)))


#Since I have ran the test data to generate 20 y predictions and error vectors- I will exporting the results to dataframe just for future reference.
df_pred.to_csv('Case2_Pred.csv', index=None)
err.to_csv('Case2_Err.csv', index=None)  

##Now it is time to calculate the correlation of errors. 
##I ran the predict function 20 times through the test data selecting the models at random each time. So for the same test data it will give us different predictions each time due to a random selection of 3 models
##I calculate the correlation of errors of the 20 times I run the test data through the predict function, show a 20*20 matrix of correlation and calculate the average correlation of errors like I did in case1

###View the correlation matrix of errors
err_corr=err.corr()
sns.set(rc={"figure.figsize":(15, 15)}) 

sns.heatmap(err_corr, annot=True)
plt.savefig('Case2_correlation_matrix.png')


##Calculating the average correlation of errors###
##summing of the matrix: upper triangular
def Upper_Sum(mat, r, c):
 
    i, j = 0, 0
    upper_sum = 0
 
    # To calculate sum of upper triangle
    for i in range(r):
        for j in range(c):
            if (i < j):
                upper_sum += (mat[i][j])
 
    return upper_sum


##denominator
def count_upper_triangular(n):
    return ((n*n)-n)/2

##Average error correlation between each iteration

def upper_sum_divided_by_cells(mat, r, c, n):
    return Upper_Sum(mat, r, c)/count_upper_triangular(n)

err_mat=err_corr.to_numpy()
print('Average error correlation :', upper_sum_divided_by_cells(err_mat, 20, 20, 20))   








