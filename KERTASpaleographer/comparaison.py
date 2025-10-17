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

# ChainCode Models
from models.chaincode_models.temp_hml_randomForest import accuracy_randomForest
from models.chaincode_models.temp_hml_svm import Accuracy_SVM
from models.chaincode_models.temp_hml_Adaboost import Accuracy_adaBoost
from models.chaincode_models.temp_hml_2_GradientBT import Accuracy_Gradient_grid

# Polygon Models
from models.polygon_models.temp_hml_poly_svm import Accuracy_SVM_Poly
from models.polygon_models.temp_hml_poly_GradientBT import Accuracy_gradientBoost_poly
from models.polygon_models.temp_hml_randomForest_poly import Accuracy_randomForest_poly


      
Classification = {
              'ChainCodeGlobal_SVM': [round(Accuracy_SVM,2)], 
              'ChainCodeGlobal_RF': [round(accuracy_randomForest,2)], 
              'ChainCodeGlobal_AdaBoost':[round(Accuracy_adaBoost,2)] ,
              'ChainCodeGlobal_Gradient': [round(Accuracy_Gradient_grid,2)],
              'Polygon_SVM':[round(Accuracy_SVM_Poly,2)], 
              'Polygon_RF':[round(Accuracy_randomForest_poly,2)],
              'Polygon_AdaBoost':[] ,
              'Polygon_GBT':[round(Accuracy_gradientBoost_poly,2)],                  }


