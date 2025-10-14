# -*- coding: utf-8 -*-
"""
Spyder Editor

'''# -*- coding: utf-8 -*-
""
Spyder Editor

This is a temporary script file.


import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

Features file X
X = pd.read_csv (r'/Users/aymen/Documents/MATLAB/NEW FEATURES (1)/Data/training/features_training_ChainCodeGlobalFE.csv')
Xt=pd.read_csv (r'/Users/aymen/Documents/MATLAB/NEW FEATURES (1)/Data/testing/features_testing_ChainCodeGlobalFE.csv')

Y = pd.read_csv (r'/Users/aymen/Documents/MATLAB/NEW FEATURES (1)/Data/training/label_training.csv')
Yt=pd.read_csv (r'/Users/aymen/Documents/MATLAB/NEW FEATURES (1)/Data/testing/label_testing.csv')
X et Xt sont le training et testing dataset 
Y et Yt sont le training et testing labels 

g=90
c=82

print('gamma is ', g , ': ' , 'and c is :', c)
#for c in np.arange (0.01, 2.02, 0.1):
Instance_SVM=SVC(kernel='rbf',C=c,gamma=g)
Instance_SVM.fit(X,Y.values.ravel()) #classification
predicted_dates=Instance_SVM.predict(Xt) 
Accuracy= accuracy_score(Yt, predicted_dates)*100 #c'est du classfication alors en peut....
print(Accuracy , '%')
'''

This is a temporary script file.
"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib as plt
import seaborn as sns
import os

# Dynamic paths - works anywhere after cloning from GitHub
script_dir = os.path.dirname(os.path.abspath(__file__))

'''Features file X'''
'''chaincode FE'''
X = pd.read_csv(os.path.join(script_dir, 'training', 'features_training_ChainCodeGlobalFE.csv'))
Xt = pd.read_csv(os.path.join(script_dir, 'testing', 'features_testing_chainCodeGlobalFE.csv'))

Y = pd.read_csv(os.path.join(script_dir, 'training', 'label_training.csv'))
Yt = pd.read_csv(os.path.join(script_dir, 'testing', 'label_testing.csv'))

print(f"✓ Data loaded | Training: {X.shape[0]} samples, Testing: {Xt.shape[0]} samples")


'''X et Xt sont le training et testing dataset 
Y et Yt sont le training et testing labels '''

'''
Section 1
g=50000
c=49999
print('gamma is ', g , ': ' , 'and c is :', c)
 #for c in np.arange (0.01, 2.02, 0.1):

'''
'''Section 2'''
# defining parameter range
param_grid = {'C': [500, 1000, 5000, 10000, 50000],
              'gamma': [50000, 5000, 500, 50, 5],
              'kernel': ['rbf']}
 
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
 
# fitting the model for grid search
grid.fit(X, Y.values.ravel())
# print best parameter after tuning
print(grid.best_params_)

# print how our model looks after hyper-parameter tuning
#print(grid.best_estimator_)

grid_predictions = grid.predict(Xt)
#print(classification_report(Yt, grid.predict(Xt)))
#Accuracy_Grid_SVM= accuracy_score(Yt, grid_predictions) #c'est du classfication alors en peut....

#print('Accuracy is : ',Accuracy_Grid_SVM , '%')
best_params_SVM=grid.best_params_#best params
#print('Classification report: ',classification_report(Yt, grid.predict(Xt)))
#print('Confusion Matrix : ',confusion_matrix(Yt, grid.predict(Xt)))
print('========= Ending grid searching =========')
#print('========= End section 2 =========')                 
Instance_SVM=SVC(kernel=best_params_SVM['kernel'],C=best_params_SVM['C'],gamma=best_params_SVM['gamma'])
Instance_SVM.fit(X,Y.values.ravel()) #classification
predicted_dates=Instance_SVM.predict(Xt) 
Accuracy_SVM= accuracy_score(Yt, predicted_dates)*100 ##Varience 
print('Accuracy is : ',Accuracy_SVM)
print('Classification report: \n',classification_report(Yt, Instance_SVM.predict(Xt)))
print('Confusion Matrix : \n',confusion_matrix(Yt, Instance_SVM.predict(Xt)))
print('========= End  =========')
# print classification report
#print(classification_report(Yt.values.ravel(), grid_predictions))
#print('gamma is ', g , ': ' , 'and c is :', c)
matrix=confusion_matrix(Yt,predicted_dates)

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
plt.pyplot.title('Confusion Matrix for SVM Model using Chaicode Global FE')
plt.pyplot.show()
print('========= End  =========')
# print classification report
classif_report=classification_report(Yt, predicted_dates)
#classif_report=matrix.astype('float')
def plot_classification_report(cr, title='Classification report for SVM model using chaincodeGlobal FE ', with_avg_total=False, cmap=plt.cm.Blues):
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
''''''''''''''''''''''''''''''
