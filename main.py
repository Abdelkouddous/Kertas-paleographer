#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning Classification Application
Main Entry Point

@author: aymen
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.ensemble import (
    RandomForestClassifier, 
    AdaBoostClassifier, 
    GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report
)
from sklearn.model_selection import GridSearchCV

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration class for data paths and model parameters"""
    
    # Data paths - Using relative paths from main.py location
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "KERTASpaleographer")
    
    # Training data paths
    TRAINING_CHAINCODE_PATH = os.path.join(DATA_DIR, "training/features_training_ChainCodeGlobalFE.csv")
    TRAINING_POLYGON_PATH = os.path.join(DATA_DIR, "training/features_training_PolygonFE.csv")
    TRAINING_LABELS_PATH = os.path.join(DATA_DIR, "training/label_training.csv")
    
    # Testing data paths
    TESTING_CHAINCODE_PATH = os.path.join(DATA_DIR, "testing/features_testing_chainCodeGlobalFE.csv")
    TESTING_POLYGON_PATH = os.path.join(DATA_DIR, "testing/features_testing_PolygonFE.csv")
    TESTING_LABELS_PATH = os.path.join(DATA_DIR, "testing/label_testing.csv")
    
    # Class names
    CLASS_NAMES = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 
                   'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14']
    
    # Model parameters
    SVM_PARAM_GRID = {
        'C': [500, 1000, 5000, 10000, 50000],
        'gamma': [50000, 5000, 500, 50, 5],
        'kernel': ['rbf']
    }
    
    RF_PARAM_GRID = {
        'n_estimators': [5, 10, 20, 100],
        'max_depth': [1, 5, 10, 20]
    }
    
    GBT_PARAM_GRID = {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': np.arange(100, 400, 100)
    }
    
    ADABOOST_PARAM_GRID = {
        "estimator__criterion": ["gini", "entropy"],
        "estimator__splitter": ["best", "random"],
        "n_estimators": [1, 2]
    }

# ============================================================================
# DATA LOADER
# ============================================================================

class DataLoader:
    """Handles loading and validation of datasets"""
    
    @staticmethod
    def load_data(feature_type='chaincode'):
        """
        Load training and testing data
        
        Args:
            feature_type (str): 'chaincode' or 'polygon'
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        try:
            if feature_type.lower() == 'chaincode':
                X_train = pd.read_csv(Config.TRAINING_CHAINCODE_PATH)
                X_test = pd.read_csv(Config.TESTING_CHAINCODE_PATH)
            elif feature_type.lower() == 'polygon':
                X_train = pd.read_csv(Config.TRAINING_POLYGON_PATH)
                X_test = pd.read_csv(Config.TESTING_POLYGON_PATH)
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")
            
            y_train = pd.read_csv(Config.TRAINING_LABELS_PATH)
            y_test = pd.read_csv(Config.TESTING_LABELS_PATH)
            
            # FIX: Reset column names to avoid feature name mismatch in scikit-learn
            # This ensures training and testing data have identical column names
            X_train.columns = range(X_train.shape[1])
            X_test.columns = range(X_test.shape[1])
            
            print(f"✓ Data loaded successfully")
            print(f"  Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
            print(f"  Testing samples: {X_test.shape[0]}, Features: {X_test.shape[1]}")
            
            return X_train, X_test, y_train, y_test
            
        except FileNotFoundError as e:
            print(f"✗ Error: Data file not found - {e}")
            print("\nPlease update the data paths in the Config class.")
            sys.exit(1)

# ============================================================================
# MODEL TRAINERS
# ============================================================================

class ModelTrainer:
    """Base class for model training and evaluation"""
    
    def __init__(self, X_train, X_test, y_train, y_test, feature_type):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train.values.ravel()
        self.y_test = y_test.values.ravel()
        self.feature_type = feature_type
        self.model = None
        self.predictions = None
        self.accuracy = None
        
    def train(self):
        """To be implemented by subclasses"""
        raise NotImplementedError
        
    def evaluate(self):
        """Evaluate model and print results"""
        self.predictions = self.model.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, self.predictions)
        
        print(f"\n{'='*60}")
        print(f"Model Accuracy: {self.accuracy:.4f} ({self.accuracy*100:.2f}%)")
        print(f"{'='*60}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, self.predictions))
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.predictions))
        
        return self.accuracy
        
    def plot_confusion_matrix(self, save_path=None):
        """Plot and optionally save confusion matrix"""
        matrix = confusion_matrix(self.y_test, self.predictions)
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(12, 10))
        sns.set(font_scale=1.2)
        sns.heatmap(matrix, annot=True, fmt='.2f', 
                   cmap=plt.cm.Blues, linewidths=0.5,
                   xticklabels=Config.CLASS_NAMES,
                   yticklabels=Config.CLASS_NAMES)
        
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(f'Confusion Matrix - {self.__class__.__name__} ({self.feature_type})', 
                 fontsize=14, pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix saved to {save_path}")
        
        plt.show()

class SVMTrainer(ModelTrainer):
    """SVM Model Trainer"""
    
    def train(self, use_grid_search=True):
        print("\n" + "="*60)
        print(f"Training SVM Model ({self.feature_type} features)")
        print("="*60)
        
        if use_grid_search:
            print("Performing Grid Search for hyperparameter tuning...")
            grid = GridSearchCV(SVC(), Config.SVM_PARAM_GRID, 
                              refit=True, verbose=2, cv=3)
            grid.fit(self.X_train, self.y_train)
            
            print(f"\nBest Parameters: {grid.best_params_}")
            print(f"Best CV Score: {grid.best_score_:.4f}")
            
            self.model = grid.best_estimator_
        else:
            self.model = SVC(kernel='rbf', C=10000, gamma=500)
            self.model.fit(self.X_train, self.y_train)
        
        print("✓ SVM training completed")
        return self

class RandomForestTrainer(ModelTrainer):
    """Random Forest Model Trainer"""
    
    def train(self, use_grid_search=True):
        print("\n" + "="*60)
        print(f"Training Random Forest Model ({self.feature_type} features)")
        print("="*60)
        
        if use_grid_search:
            print("Performing Grid Search for hyperparameter tuning...")
            grid_search = GridSearchCV(RandomForestClassifier(), 
                                      param_grid=Config.RF_PARAM_GRID, 
                                      cv=3, n_jobs=-1, verbose=1)
            grid_search.fit(self.X_train, self.y_train)
            
            print(f"\nBest Parameters: {grid_search.best_params_}")
            print(f"Best CV Score: {grid_search.best_score_:.4f}")
            
            self.model = grid_search.best_estimator_
        else:
            self.model = RandomForestClassifier(n_estimators=100, max_depth=20)
            self.model.fit(self.X_train, self.y_train)
        
        print("✓ Random Forest training completed")
        return self

class GradientBoostingTrainer(ModelTrainer):
    """Gradient Boosting Model Trainer"""
    
    def train(self, use_grid_search=True):
        print("\n" + "="*60)
        print(f"Training Gradient Boosting Model ({self.feature_type} features)")
        print("="*60)
        
        if use_grid_search:
            print("Performing Grid Search for hyperparameter tuning...")
            gb = GradientBoostingClassifier()
            gb_cv = GridSearchCV(gb, Config.GBT_PARAM_GRID, cv=4, verbose=1)
            gb_cv.fit(self.X_train, self.y_train)
            
            print(f"\nBest Parameters: {gb_cv.best_params_}")
            print(f"Train Score: {gb_cv.best_score_:.4f}")
            
            self.model = gb_cv.best_estimator_
        else:
            self.model = GradientBoostingClassifier(n_estimators=300, 
                                                   learning_rate=0.1, 
                                                   random_state=100)
            self.model.fit(self.X_train, self.y_train)
        
        print("✓ Gradient Boosting training completed")
        return self

class AdaBoostTrainer(ModelTrainer):
    """AdaBoost Model Trainer"""
    
    def train(self, use_grid_search=False):
        print("\n" + "="*60)
        print(f"Training AdaBoost Model ({self.feature_type} features)")
        print("="*60)
        
        DTC = DecisionTreeClassifier(random_state=11,
                                     max_features=0.8,
                                     class_weight="balanced",
                                     max_depth=None)
        
        if use_grid_search:
            print("Performing Grid Search for hyperparameter tuning...")
            ABC = AdaBoostClassifier(estimator=DTC)
            grid_search = GridSearchCV(ABC, 
                                      param_grid=Config.ADABOOST_PARAM_GRID,
                                      scoring='accuracy', cv=3, verbose=1)
            grid_search.fit(self.X_train, self.y_train)
            
            print(f"\nBest Parameters: {grid_search.best_params_}")
            print(f"Best CV Score: {grid_search.best_score_:.4f}")
            
            self.model = grid_search.best_estimator_
        else:
            self.model = AdaBoostClassifier(estimator=DTC)
            self.model.fit(self.X_train, self.y_train)
        
        print("✓ AdaBoost training completed")
        return self

# ============================================================================
# MODEL COMPARISON
# ============================================================================

class ModelComparison:
    """Compare multiple models on the same dataset"""
    
    def __init__(self, feature_type='chaincode'):
        self.feature_type = feature_type
        self.results = {}
        
    def run_comparison(self, use_grid_search=True):
        """Run all models and compare results"""
        print("\n" + "="*70)
        print(f"MODEL COMPARISON - {self.feature_type.upper()} FEATURES")
        print("="*70)
        
        # Load data
        X_train, X_test, y_train, y_test = DataLoader.load_data(self.feature_type)
        
        # Define models to compare
        models = {
            'SVM': SVMTrainer,
            'Random Forest': RandomForestTrainer,
            'Gradient Boosting': GradientBoostingTrainer,
            'AdaBoost': AdaBoostTrainer
        }
        
        # Train and evaluate each model
        for model_name, ModelClass in models.items():
            try:
                trainer = ModelClass(X_train, X_test, y_train, y_test, self.feature_type)
                trainer.train(use_grid_search=use_grid_search)
                accuracy = trainer.evaluate()
                self.results[model_name] = {
                    'accuracy': accuracy,
                    'trainer': trainer
                }
            except Exception as e:
                print(f"\n✗ Error training {model_name}: {e}")
                self.results[model_name] = {'accuracy': 0, 'trainer': None}
        
        # Print comparison summary
        self.print_summary()
        
        return self.results
    
    def print_summary(self):
        """Print comparison summary"""
        print("\n" + "="*70)
        print(f"SUMMARY - {self.feature_type.upper()} FEATURES")
        print("="*70)
        
        # Sort by accuracy
        sorted_results = sorted(self.results.items(), 
                               key=lambda x: x[1]['accuracy'], 
                               reverse=True)
        
        print(f"\n{'Model':<25} {'Accuracy':<15} {'Percentage':<15}")
        print("-" * 55)
        
        for model_name, data in sorted_results:
            accuracy = data['accuracy']
            print(f"{model_name:<25} {accuracy:<15.4f} {accuracy*100:<15.2f}%")
        
        # Best model
        best_model = sorted_results[0]
        print("\n" + "="*70)
        print(f"🏆 Best Model: {best_model[0]} with {best_model[1]['accuracy']*100:.2f}% accuracy")
        print("="*70)
    
    def plot_comparison(self, save_path=None):
        """Plot accuracy comparison bar chart"""
        models = list(self.results.keys())
        accuracies = [self.results[m]['accuracy'] * 100 for m in models]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%', ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title(f'Model Comparison - {self.feature_type.upper()} Features', 
                 fontsize=14, pad=20)
        plt.ylim(0, 100)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Comparison plot saved to {save_path}")
        
        plt.show()

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def print_menu():
    """Print main menu"""
    print("\n" + "="*70)
    print("  MACHINE LEARNING CLASSIFICATION SYSTEM")
    print("="*70)
    print("\n1. Train Single Model")
    print("2. Compare All Models")
    print("3. Train on ChainCode Features")
    print("4. Train on Polygon Features")
    print("5. Compare Both Feature Types")
    print("6. Exit")
    print("\n" + "-"*70)

def train_single_model():
    """Train a single model selected by user"""
    print("\nSelect Model:")
    print("1. SVM")
    print("2. Random Forest")
    print("3. Gradient Boosting")
    print("4. AdaBoost")
    
    model_choice = input("\nEnter choice (1-4): ").strip()
    
    print("\nSelect Feature Type:")
    print("1. ChainCode")
    print("2. Polygon")
    
    feature_choice = input("\nEnter choice (1-2): ").strip()
    feature_type = 'chaincode' if feature_choice == '1' else 'polygon'
    
    use_grid = input("\nUse Grid Search? (y/n): ").strip().lower() == 'y'
    
    # Load data
    X_train, X_test, y_train, y_test = DataLoader.load_data(feature_type)
    
    # Select and train model
    model_map = {
        '1': SVMTrainer,
        '2': RandomForestTrainer,
        '3': GradientBoostingTrainer,
        '4': AdaBoostTrainer
    }
    
    if model_choice in model_map:
        ModelClass = model_map[model_choice]
        trainer = ModelClass(X_train, X_test, y_train, y_test, feature_type)
        trainer.train(use_grid_search=use_grid)
        trainer.evaluate()
        
        # Ask if user wants to plot confusion matrix
        plot = input("\nPlot confusion matrix? (y/n): ").strip().lower() == 'y'
        if plot:
            trainer.plot_confusion_matrix()
    else:
        print("Invalid model choice!")

def main():
    """Main entry point"""
    while True:
        print_menu()
        choice = input("Enter your choice (1-6): ").strip()
        
        if choice == '1':
            train_single_model()
            
        elif choice == '2':
            feature_type = input("\nEnter feature type (chaincode/polygon): ").strip().lower()
            use_grid = input("Use Grid Search? (y/n): ").strip().lower() == 'y'
            
            comparison = ModelComparison(feature_type)
            comparison.run_comparison(use_grid_search=use_grid)
            
            plot = input("\nPlot comparison chart? (y/n): ").strip().lower() == 'y'
            if plot:
                comparison.plot_comparison()
                
        elif choice == '3':
            use_grid = input("\nUse Grid Search? (y/n): ").strip().lower() == 'y'
            comparison = ModelComparison('chaincode')
            comparison.run_comparison(use_grid_search=use_grid)
            comparison.plot_comparison()
            
        elif choice == '4':
            use_grid = input("\nUse Grid Search? (y/n): ").strip().lower() == 'y'
            comparison = ModelComparison('polygon')
            comparison.run_comparison(use_grid_search=use_grid)
            comparison.plot_comparison()
            
        elif choice == '5':
            use_grid = input("\nUse Grid Search? (y/n): ").strip().lower() == 'y'
            
            print("\n" + "="*70)
            print("COMPARING BOTH FEATURE TYPES")
            print("="*70)
            
            # ChainCode
            comp1 = ModelComparison('chaincode')
            results1 = comp1.run_comparison(use_grid_search=use_grid)
            
            # Polygon
            comp2 = ModelComparison('polygon')
            results2 = comp2.run_comparison(use_grid_search=use_grid)
            
            # Overall comparison
            print("\n" + "="*70)
            print("FEATURE TYPE COMPARISON")
            print("="*70)
            
            for model in results1.keys():
                acc1 = results1[model]['accuracy'] * 100
                acc2 = results2[model]['accuracy'] * 100
                print(f"\n{model}:")
                print(f"  ChainCode: {acc1:.2f}%")
                print(f"  Polygon:   {acc2:.2f}%")
                print(f"  Best: {'ChainCode' if acc1 > acc2 else 'Polygon'}")
            
        elif choice == '6':
            print("\n✓ Exiting... Goodbye!")
            break
            
        else:
            print("\n✗ Invalid choice! Please enter 1-6.")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║     MACHINE LEARNING CLASSIFICATION APPLICATION                  ║
    ║     Author: Aymen                                                ║
    ║     Version: 1.0                                                 ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Check if running in interactive mode or with arguments
    if len(sys.argv) > 1:
        # Command line mode (future enhancement)
        print("Command line mode not yet implemented. Please run interactively.")
    else:
        # Interactive mode
        main()

