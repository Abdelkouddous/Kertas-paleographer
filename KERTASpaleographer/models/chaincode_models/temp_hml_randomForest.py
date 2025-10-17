#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 00:53:31 2022

@author: aymen
"""

import pandas as pd
#import numpy as np
# Import the model we are using
from sklearn.metrics import  confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib as plt
import seaborn as sns
import os

# Dynamic paths - works anywhere after cloning from GitHub
# Go up two levels: from models/chaincode_models/ to KERTASpaleographer/
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#Accuracy score do not work on regression
'''Features file X'''
X = pd.read_csv(os.path.join(script_dir, 'training', 'features_training_ChainCodeGlobalFE.csv'))
Xt = pd.read_csv(os.path.join(script_dir, 'testing', 'features_testing_chainCodeGlobalFE.csv'))

Y = pd.read_csv(os.path.join(script_dir, 'training', 'label_training.csv'))
Yt = pd.read_csv(os.path.join(script_dir, 'testing', 'label_testing.csv'))

print(f"✓ Data loaded | Training: {X.shape[0]} samples, Testing: {Xt.shape[0]} samples")
'''X et Xt sont le training et testing dataset 
Y et Yt sont le training et testing labels '''


# Instantiate model with 1000 decision trees estimators is the number of elemnts of the data
#for estimator in range (100,105,1):

params = {
    'n_estimators': [5, 10, 20,100],
    'max_depth': [1, 5, 10,20],
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid=params, n_jobs=-1)
grid_search.fit(X, Y.values.ravel())
#score_randomForest_grid = np.round(grid_search.score(Xt, Yt), 3)
#print(f"Best random forest classifier score: {score_randomForest_grid}" )
best_params_RF=grid_search.best_params_
best_estimator_RF=grid_search.best_estimator_
print('Using the RF algorithm')
#Accuracy_randomForest_grid=accuracy_score(Yt,grid_search.predict(Xt))
rf = RandomForestClassifier(n_estimators = best_params_RF['n_estimators'], 
                           max_depth=best_params_RF['max_depth'])
rf.fit(X,Y.values.ravel())
predictions_RF=rf.predict(Xt)
accuracy_randomForest = accuracy_score(Yt, predictions_RF) #Varience 
print('best score is : ' , accuracy_randomForest, '\n')
print(classification_report(Yt, predictions_RF))
print(confusion_matrix(Yt,predictions_RF))

'''random state est plus lourd quand on l'augemente mais le taux d'err diminue
c'est pas le cas pour l'estmator '''
'''plotting'''
matrix=confusion_matrix(Yt,predictions_RF)
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
plt.pyplot.title('Confusion Matrix for RandomForest Model using Chaicode Global FE')
plt.pyplot.show()
print('========= End  =========')
# print classification report
classif_report=classification_report(Yt, predictions_RF)
#classif_report=matrix.astype('float')
def plot_classification_report(cr, title='Classification report for RandomForest model using ChaincodeGlobal FE ', with_avg_total=False, cmap=plt.cm.Blues):
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