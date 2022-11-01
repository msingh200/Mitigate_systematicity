#Importing all the necessary libraries

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
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import random
import pickle
import shutil
import os
import sys







##Function for processing the data
##'candidate_id', 'occupation_id', 'company_id', 'application_attribute_1' are dropped since these features have no duplication of values, I would have to include 49999 dummies in the model which would help predict nothing. 
##However, in real-world I would not drop these covariates if there was much consideration duplication such that generating dummies is meaningful
##Application_status is the dependent variable
##Hired=1, all other categories=0 for the dependent variable generated from application status
##Normalize the data subtracting mean and dividing std deviation
##generate dummy variable dropping one category for ‘candidate_demographic_variable_5’
##Also drop gender and ethnicity since these features might not have any bearing on our predictions


def preprocessingPipeline(dataFileName):
    def importData(filename):
        df = pd.read_csv(filename)
        return df

    def dataCleaning(df_org):
        #Droping useless features
        df = df_org.drop(['candidate_id', 'occupation_id', 'company_id', 'application_attribute_1', 'ethnicity', 'gender'], axis=1)

        #filling NaN values
        df = df.fillna(0)

        #Encoding Lable
        df['application_status'] = df['application_status'].replace('hired', 1)
        df['application_status'] = df['application_status'].replace(['pre-interview', 'interview'], 0)

        

        df_y = df['application_status']
        df = df.drop(['application_status'], axis=1)

        #One Hot Encoding (candidate_demographic_variable_5)
        df = pd.get_dummies(df, drop_first=True)

        #normalization
        df = preprocessing.normalize(df)

        return df, df_y

    df_org = importData(dataFileName)
    X, y = dataCleaning(df_org)
    return X, y







###I TRAIN 10 different models and report the accuracy, precision, recall and f1_score on each of these models. Here is the function for doing so:


def train(X, y):
    weights= {0:0.72, 1:1.64}
    def modelTraining(model, modelName):
        model.fit(x_train, y_train)
        
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        #Exporting pred data.
        np.save('Test Data/y_test_pred {}'.format(modelName), y_test_pred)

        print('\n____ :{}:____'.format(modelName))
        #accuracy checking
        
        print('Accuracy train data = ', format(accuracy_score(y_train, y_train_pred)))
        print('Accuracy test data = ', format(accuracy_score(y_test, y_test_pred)))
        
        print('F1 train data = ', format(f1_score(y_train, y_train_pred, average='binary')))
        print('F1 test data = ', format(f1_score(y_test, y_test_pred, average='binary')))
        
        print('Precision train data = ', format(precision_score(y_train, y_train_pred, average='binary')))
        print('Precision test data = ', format(precision_score(y_test, y_test_pred, average='binary')))
        
        print('Recall train data = ', format(recall_score(y_train, y_train_pred, average='binary')))
        print('Recall test data = ', format(recall_score(y_test, y_test_pred, average='binary')))

        filename = 'Models/{}.sav'.format(modelName)
        pickle.dump(model, open(filename, 'wb'))

    # Making Dirs
    try:
        path  = 'Models'
        shutil.rmtree(path)
        os.mkdir(path)
    except:
        os.mkdir(path)
    
    try:
        path  = 'Test Data'
        shutil.rmtree(path)
        os.mkdir(path)
    except:
        os.mkdir(path)

    #Train test split (70-30)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    np.save('Test Data/X_test', x_test)
    np.save('Test Data/y_test', y_test)    

    # Training 
    model=SVC(gamma=0.1, C=1000, class_weight=weights)
    modelTraining(model, 'SVM')

    model = LinearSVC(tol=1e-5, class_weight=weights)
    modelTraining(model, 'SVM Linear')

    model = tree.DecisionTreeClassifier(max_depth=10, class_weight=weights)
    modelTraining(model, 'Decision Tree')

    model = RandomForestClassifier(n_estimators=30, min_samples_split=180, class_weight=weights)
    modelTraining(model, 'Random Forest')

    model = KNeighborsClassifier(n_neighbors=10)
    modelTraining(model, 'KNN')

    model = ComplementNB()
    modelTraining(model, 'Naive Bayes')
    
    model = LogisticRegression(C=10, class_weight=weights)
    modelTraining(model, 'Logistic Regression')
    
    model = AdaBoostClassifier(n_estimators=100)
    modelTraining(model, 'Adaboost Classifier')

    model = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(16, 6), random_state=1)
    modelTraining(model, 'MLP Classifier')

    model = QuadraticDiscriminantAnalysis()
    modelTraining(model, 'QDA')








##I call the preprocessing function on dataset and train function so that I train the 10 different models and report the evaluation metrics on each.
##When running the models, I adjust for class_weights. Adjusting with weights is often a better method than creating synthetic data/undersampling.
##KNN is not affected by class imbalance so weights not adjusted
##ComplementNB performs well in cases of imbalanced data and handles the problem of class imbalance inherently as it calculates the probability of belonging to all classes
##Adaboost also by default handles class imbalance problem well by maintaining a set of weights on the training data set in the learning process

X, y=preprocessingPipeline('data.csv')

train(X,y)




