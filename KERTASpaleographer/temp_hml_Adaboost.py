#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 02:12:41 2022

@author: aymen
"""

from sklearn.ensemble import AdaBoostClassifier
#from sklearn.ensemble import AdaBoostRegressor
#Creating decision tree stump model
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
#importing plotting packages
import numpy as np
import matplotlib as plt
import seaborn as sns
import os

# Dynamic paths - works anywhere after cloning from GitHub
script_dir = os.path.dirname(os.path.abspath(__file__))

X = pd.read_csv(os.path.join(script_dir, 'training', 'features_training_ChainCodeGlobalFE.csv'))
Xt = pd.read_csv(os.path.join(script_dir, 'testing', 'features_testing_chainCodeGlobalFE.csv'))

Y = pd.read_csv(os.path.join(script_dir, 'training', 'label_training.csv'))
Yt = pd.read_csv(os.path.join(script_dir, 'testing', 'label_testing.csv'))

print(f"✓ Data loaded | Training: {X.shape[0]} samples, Testing: {Xt.shape[0]} samples")

wine = load_wine()

# run grid search
param_grid = {"estimator__criterion" : ["gini", "entropy"],
              "estimator__splitter" :   ["best", "random"],
              "n_estimators": [1, 2]
             }


DTC = DecisionTreeClassifier(random_state = 11,
                             max_features = "auto",
                             class_weight = "balanced",
                             max_depth = None)

ABC = AdaBoostClassifier(estimator = DTC)
#grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')
grid_search_ABC = GridSearchCV(ABC, param_grid=param_grid,scoring = 'accuracy')
grid_search_ABC.fit(X,Y.values.ravel()) #Classification
predicted_ABC=grid_search_ABC.predict(Xt)
#accuracy_adaBoost_grid = accuracy_score(Yt, predicted_ABC)
#best_params_ABC=grid_search_ABC.best_params_
#print("Best: %f using %s" % (grid_search_ABC.best_score_ , best_params_ABC))
#adaclf_train_sc = accuracy_score(Y, ABC.predict(X))
#print('classification report: ',classification_report(Yt, ABC.predict(Xt)), '===')
#print('AdaBoost  accuracies %.3f' % (Accuracy_adaBoost_grid))

adaclf = AdaBoostClassifier(DTC)
adaclf.fit(X, Y.values.ravel())
predicted=adaclf.predict(Xt)
Accuracy_adaBoost = accuracy_score(Yt, predicted)
print(classification_report(Yt, predicted))
print('Classification report :\n ', classification_report(Yt, predicted))
print( 'Adaboost confusion matrix : \n' , confusion_matrix(Yt,predicted))
print('Adaptive boosting tree accuracies %.3f' % (Accuracy_adaBoost))
#Create an AdaBoost classification model


#Raveling

matrix=confusion_matrix(Yt,predicted)
#configuring the matrix 
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
# Build the plot
plt.figure.Figure(figsize=(14,14))
sns.set(font_scale=1)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.4)

# Add labels to the plot
class_names = ['C1', ' C2', 'C3', 
               'C4', 'C5','C6',
               'C7','C8','C9',
               'C10','C11','C12',
               'C13','C14',]
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + .5
#drawing the table
plt.pyplot.xticks(tick_marks, class_names, rotation=90)
plt.pyplot.yticks(tick_marks2, class_names, rotation=0)
plt.pyplot.xlabel('Predicted label')
plt.pyplot.ylabel('True label')
plt.pyplot.subplots_adjust(left=0.2, bottom=0.2)
plt.pyplot.title('Confusion Matrix for Adaboost Model using Chaicode Global FE')
plt.pyplot.show()
print('========= End  =========')
# print classification report
classif_report=classification_report(Yt, predicted)
#classif_report=matrix.astype('float')
def plot_classification_report(cr, title='Classification report for Adaboost model using ChaincodeGlobal FE ', with_avg_total=False, cmap=plt.cm.Blues):
    lines = cr.split('\n')
    classes = []
    plotMat = []
    for line in lines[2 : (len(lines) - 6)]: 
        t = line.split()
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        plotMat.append(v)
    if with_avg_total:
        aveTotal = lines[len(lines) - 1].split()
        classes.append('avg/total')
        vAveTotal = [float(x) for x in t[1:len(aveTotal) - 1]]
        plotMat.append(vAveTotal)
    plt.pyplot.imshow(plotMat, interpolation='nearest', cmap=cmap)
    plt.pyplot.title(title)
    ##plt.figure.Figure(figsize=(18,20))##'''
    y_classes = ['C1', ' C2', 'C3', 
                   'C4', 'C5','C6',
                   'C7','C8','C9',
                   'C10','C11','C12',
                   'C13','C14',]
    x_axisClass= ['precision', 'recall', 'f1-score']
    x_tick_marks = np.arange(len(x_axisClass))
    y_tick_marks = np.arange(len(y_classes)) #np.arange(len(classes))
    plt.pyplot.xticks(x_tick_marks, x_axisClass, rotation=90)
    plt.pyplot.yticks(y_tick_marks, y_classes, rotation=0)
    plt.pyplot.tight_layout()
    plt.pyplot.ylabel('Classes')
    plt.pyplot.xlabel('Measures')
    #sns.heatmap(plotMat, annot=True,annot_kws={'size':10})
    #plt.pyplot.subplots_adjust(left=0.2, bottom=0.2)
    ##sns.heatmap(plotMat, annot=True) ##
plot_classification_report(classif_report, with_avg_total=True)


