#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 12:25:41 2022

@author: aymen abdelkouddous hamel
GBT
"""
# Import all relevant libraries

from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#import warnings
from sklearn.model_selection import GridSearchCV
#
import matplotlib as plt
import seaborn as sns
import os

# Get the directory where this script is located
# Go up two levels: from models/chaincode_models/ to KERTASpaleographer/
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Build paths relative to script location
feature_training_data_path = os.path.join(script_dir, 'training', 'features_training_ChainCodeGlobalFE.csv')
# Note: testing file has lowercase 'c' in chainCode
feature_testing_data_path = os.path.join(script_dir, 'testing', 'features_testing_chainCodeGlobalFE.csv')

label_training_path = os.path.join(script_dir, 'training', 'label_training.csv')
label_testing_path = os.path.join(script_dir, 'testing', 'label_testing.csv')

print(f"Loading training features from: {feature_training_data_path}")
print(f"Loading testing features from: {feature_testing_data_path}")

X = pd.read_csv(feature_training_data_path)
Xt = pd.read_csv(feature_testing_data_path)

Y = pd.read_csv(label_training_path)
Yt = pd.read_csv(label_testing_path)

print(f"✓ Data loaded successfully!")
print(f"  Training samples: {X.shape[0]}, Features: {X.shape[1]}")
print(f"  Testing samples: {Xt.shape[0]}, Features: {Xt.shape[1]}")

#warnings.filterwarnings("ignore")

'''section 1 grid searching'''

'''grid = { 
    #To add max depths

    'learning_rate':[0.01,0.05,0.1],

    'n_estimators':np.arange(100,400,100),

}

print('Processing grid search ... ')

gb = GradientBoostingClassifier()

gb_cv = GridSearchCV(gb, grid, cv = 4) #cv=4 default

gb_cv.fit(X, Y.values.ravel())

print("Best Parameters:",gb_cv.best_params_)

print("Train Score:",gb_cv.best_score_)

print("Test Score:",gb_cv.score(Xt,Yt))

Accuracy_Gradient_grid=accuracy_score(Yt.values.ravel(), gb_cv.predict(Xt))

'''

#best_params_GBT=gb_cv.best_params_

gbc=GradientBoostingClassifier(n_estimators=300,learning_rate=0.1,random_state=100 )

# Fit train data to GBC

gbc.fit(X, Y.values.ravel())
Accuracy_gradientBoost =accuracy_score(Yt, gbc.predict(Xt))

print("GBC accuracy is %2.2f " % (Accuracy_gradientBoost))

print('Confusion matrix; ', confusion_matrix(Yt, gbc.predict(Xt)))

print('Classification report: ' , classification_report(Yt, gbc.predict(Xt)))

'''plotting
'''
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
