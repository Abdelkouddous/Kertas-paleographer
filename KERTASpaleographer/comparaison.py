#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 22:39:23 2022

@author: aymen
Comparaison
"""

# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from temp_hml_randomForest import accuracy_randomForest
from temp_hml_svm import Accuracy_SVM
from temp_hml_Adaboost import Accuracy_adaBoost

from temp_hml_poly_svm import Accuracy_SVM_Poly
from temp_hlm_GradientBT import Accuracy_Gradient_grid
      
Classification = {
                'ChainCodeGlobal_SVM': [round(Accuracy_SVM,2)], 'ChainCodeGlobal_RF': [round(accuracy_randomForest,2)], 'ChainCodeGlobal_AdaBoost':[round(Accuracy_adaBoost,2)] ,
              'ChainCodeGlobal_Gradient': [round(Accuracy_Gradient,2)],
              'Polygon_SVM':[round(Accuracy_SVM_Poly,2)], 'Polygon_RF':[0],'Polygon_AdaBoost':[] ,'Polygon_GBT':[],
              'F3_C3':[], 'F3_C3':[]
                  }


