#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Testing Suite for ML Classification System

@author: aymen abdelkouddous hamel
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from main module
from main import Config, DataLoader

class TestDataLoading(unittest.TestCase):
    """Test data loading functionality"""
    
    def test_config_paths_exist(self):
        """Test that all configured paths are valid"""
        self.assertTrue(os.path.exists(Config.DATA_DIR), "Data directory not found")
        self.assertTrue(os.path.exists(Config.TRAINING_CHAINCODE_PATH), "Training ChainCode file not found")
        self.assertTrue(os.path.exists(Config.TRAINING_POLYGON_PATH), "Training Polygon file not found")
        self.assertTrue(os.path.exists(Config.TESTING_CHAINCODE_PATH), "Testing ChainCode file not found")
        self.assertTrue(os.path.exists(Config.TESTING_POLYGON_PATH), "Testing Polygon file not found")
    
    def test_load_chaincode_data(self):
        """Test loading ChainCode features"""
        X_train, X_test, y_train, y_test = DataLoader.load_data('chaincode')
        
        # Check data loaded
        self.assertIsNotNone(X_train, "Training features not loaded")
        self.assertIsNotNone(X_test, "Testing features not loaded")
        self.assertIsNotNone(y_train, "Training labels not loaded")
        self.assertIsNotNone(y_test, "Testing labels not loaded")
        
        # Check shapes match
        self.assertEqual(X_train.shape[1], X_test.shape[1], "Feature dimensions don't match")
        self.assertEqual(len(X_train), len(y_train), "Training samples mismatch")
        self.assertEqual(len(X_test), len(y_test), "Testing samples mismatch")
        
        # Check column names are reset (fix applied)
        self.assertEqual(list(X_train.columns), list(range(X_train.shape[1])), 
                        "Training columns not reset to integers")
        self.assertEqual(list(X_test.columns), list(range(X_test.shape[1])), 
                        "Testing columns not reset to integers")
    
    def test_load_polygon_data(self):
        """Test loading Polygon features"""
        X_train, X_test, y_train, y_test = DataLoader.load_data('polygon')
        
        # Check data loaded
        self.assertIsNotNone(X_train, "Training features not loaded")
        self.assertIsNotNone(X_test, "Testing features not loaded")
        
        # Check shapes
        self.assertGreater(X_train.shape[0], 0, "No training samples")
        self.assertGreater(X_test.shape[0], 0, "No testing samples")
        self.assertGreater(X_train.shape[1], 0, "No features")

class TestDataQuality(unittest.TestCase):
    """Test data quality and integrity"""
    
    @classmethod
    def setUpClass(cls):
        """Load data once for all tests"""
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = DataLoader.load_data('chaincode')
    
    def test_no_missing_values(self):
        """Test that data has no missing values"""
        self.assertFalse(self.X_train.isnull().any().any(), "Training features have missing values")
        self.assertFalse(self.X_test.isnull().any().any(), "Testing features have missing values")
        self.assertFalse(self.y_train.isnull().any().any(), "Training labels have missing values")
        self.assertFalse(self.y_test.isnull().any().any(), "Testing labels have missing values")
    
    def test_label_values_valid(self):
        """Test that labels are within expected range"""
        unique_labels = np.unique(self.y_train)
        self.assertGreater(len(unique_labels), 1, "Only one class in training labels")
        self.assertLessEqual(len(unique_labels), 14, "Too many classes (expected max 14)")
    
    def test_feature_types(self):
        """Test that features are numeric"""
        self.assertTrue(np.issubdtype(self.X_train.values.dtype, np.number), 
                       "Training features are not numeric")
        self.assertTrue(np.issubdtype(self.X_test.values.dtype, np.number), 
                       "Testing features are not numeric")
    
    def test_data_shapes_reasonable(self):
        """Test that data shapes are reasonable"""
        # Expected approximate values based on your data
        self.assertGreater(self.X_train.shape[0], 1000, "Too few training samples")
        self.assertGreater(self.X_test.shape[0], 500, "Too few testing samples")
        self.assertGreater(self.X_train.shape[1], 100, "Too few features")

class TestModelTraining(unittest.TestCase):
    """Test model training functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Load data and prepare for model training"""
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = DataLoader.load_data('chaincode')
        cls.y_train = cls.y_train.values.ravel()
        cls.y_test = cls.y_test.values.ravel()
        
        # Use small subset for faster testing
        cls.X_train_small = cls.X_train[:100]
        cls.y_train_small = cls.y_train[:100]
        cls.X_test_small = cls.X_test[:50]
        cls.y_test_small = cls.y_test[:50]
    
    def test_random_forest_training(self):
        """Test Random Forest model training"""
        model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
        model.fit(self.X_train_small, self.y_train_small)
        predictions = model.predict(self.X_test_small)
        accuracy = accuracy_score(self.y_test_small, predictions)
        
        self.assertGreater(accuracy, 0.0, "Model accuracy is zero")
        self.assertLessEqual(accuracy, 1.0, "Model accuracy exceeds 100%")
    
    def test_svm_training(self):
        """Test SVM model training"""
        model = SVC(kernel='rbf', C=100, gamma=1, random_state=42)
        model.fit(self.X_train_small, self.y_train_small)
        predictions = model.predict(self.X_test_small)
        accuracy = accuracy_score(self.y_test_small, predictions)
        
        self.assertGreater(accuracy, 0.0, "Model accuracy is zero")
    
    def test_gradient_boosting_training(self):
        """Test Gradient Boosting model training"""
        model = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, random_state=42)
        model.fit(self.X_train_small, self.y_train_small)
        predictions = model.predict(self.X_test_small)
        accuracy = accuracy_score(self.y_test_small, predictions)
        
        self.assertGreater(accuracy, 0.0, "Model accuracy is zero")
    
    def test_adaboost_training(self):
        """Test AdaBoost model training"""
        base = DecisionTreeClassifier(max_depth=3, random_state=42)
        model = AdaBoostClassifier(estimator=base, n_estimators=10, random_state=42)
        model.fit(self.X_train_small, self.y_train_small)
        predictions = model.predict(self.X_test_small)
        accuracy = accuracy_score(self.y_test_small, predictions)
        
        self.assertGreater(accuracy, 0.0, "Model accuracy is zero")

class TestModelPredictions(unittest.TestCase):
    """Test model prediction functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Train a simple model for testing predictions"""
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = DataLoader.load_data('chaincode')
        cls.y_train = cls.y_train.values.ravel()
        cls.y_test = cls.y_test.values.ravel()
        
        # Train a simple model
        cls.model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
        cls.model.fit(cls.X_train[:100], cls.y_train[:100])
    
    def test_prediction_shape(self):
        """Test that predictions have correct shape"""
        predictions = self.model.predict(self.X_test[:10])
        self.assertEqual(len(predictions), 10, "Prediction count mismatch")
    
    def test_prediction_values(self):
        """Test that predictions are valid class labels"""
        predictions = self.model.predict(self.X_test[:10])
        unique_train_labels = np.unique(self.y_train)
        
        for pred in predictions:
            self.assertIn(pred, unique_train_labels, f"Invalid prediction: {pred}")
    
    def test_prediction_consistency(self):
        """Test that same input gives same prediction"""
        pred1 = self.model.predict(self.X_test[:5])
        pred2 = self.model.predict(self.X_test[:5])
        
        np.testing.assert_array_equal(pred1, pred2, "Predictions are not consistent")

class TestConfiguration(unittest.TestCase):
    """Test configuration settings"""
    
    def test_class_names_count(self):
        """Test that class names are correctly defined"""
        self.assertEqual(len(Config.CLASS_NAMES), 14, "Expected 14 class names")
    
    def test_param_grids_valid(self):
        """Test that parameter grids are valid"""
        # SVM params
        self.assertIn('C', Config.SVM_PARAM_GRID)
        self.assertIn('gamma', Config.SVM_PARAM_GRID)
        self.assertIn('kernel', Config.SVM_PARAM_GRID)
        
        # RF params
        self.assertIn('n_estimators', Config.RF_PARAM_GRID)
        self.assertIn('max_depth', Config.RF_PARAM_GRID)
        
        # GBT params
        self.assertIn('learning_rate', Config.GBT_PARAM_GRID)
        self.assertIn('n_estimators', Config.GBT_PARAM_GRID)
    
    def test_base_dir_valid(self):
        """Test that base directory is valid"""
        self.assertTrue(os.path.isdir(Config.BASE_DIR), "Base directory not found")

class TestFeatureNameFix(unittest.TestCase):
    """Test that the feature name mismatch fix is applied"""
    
    def test_column_names_are_integers(self):
        """Test that column names are reset to integers"""
        X_train, X_test, _, _ = DataLoader.load_data('chaincode')
        
        # Check training data
        train_cols = list(X_train.columns)
        self.assertEqual(train_cols, list(range(len(train_cols))), 
                        "Training columns are not integers")
        
        # Check testing data
        test_cols = list(X_test.columns)
        self.assertEqual(test_cols, list(range(len(test_cols))), 
                        "Testing columns are not integers")
    
    def test_column_names_match(self):
        """Test that training and testing column names match"""
        X_train, X_test, _, _ = DataLoader.load_data('chaincode')
        
        self.assertEqual(list(X_train.columns), list(X_test.columns), 
                        "Training and testing column names don't match")

# ============================================================================
# TEST RUNNER
# ============================================================================

def run_tests(verbose=True):
    """Run all tests and return results"""
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestDataQuality))
    suite.addTests(loader.loadTestsFromTestCase(TestModelTraining))
    suite.addTests(loader.loadTestsFromTestCase(TestModelPredictions))
    suite.addTests(loader.loadTestsFromTestCase(TestConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureNameFix))
    
    # Run tests
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result

def print_summary(result):
    """Print test summary"""
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║     AUTOMATED TESTING SUITE                                      ║
    ║     Machine Learning Classification System                       ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Run tests
    result = run_tests(verbose=True)
    
    # Print summary
    success = print_summary(result)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

