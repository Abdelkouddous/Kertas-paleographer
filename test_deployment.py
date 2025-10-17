#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deployment Test Script
Tests if all imports work correctly for Streamlit Cloud deployment
"""

def test_imports():
    """Test all required imports"""
    try:
        print("Testing imports...")
        
        # Test basic imports
        import streamlit as st
        print("✅ streamlit")
        
        import pandas as pd
        print("✅ pandas")
        
        import numpy as np
        print("✅ numpy")
        
        import matplotlib.pyplot as plt
        print("✅ matplotlib")
        
        import seaborn as sns
        print("✅ seaborn")
        
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
        print("✅ sklearn.ensemble")
        
        from sklearn.svm import SVC
        print("✅ sklearn.svm")
        
        from sklearn.tree import DecisionTreeClassifier
        print("✅ sklearn.tree")
        
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
        print("✅ sklearn.metrics")
        
        from sklearn.model_selection import GridSearchCV
        print("✅ sklearn.model_selection")
        
        # Test plotly imports
        import plotly.graph_objects as go
        print("✅ plotly.graph_objects")
        
        import plotly.express as px
        print("✅ plotly.express")
        
        from plotly.subplots import make_subplots
        print("✅ plotly.subplots")
        
        # Test cost function visualization
        from cost_function_visualization import CostFunctionVisualizer, generate_sample_data
        print("✅ cost_function_visualization")
        
        print("\n🎉 All imports successful! Ready for deployment.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_cost_function():
    """Test cost function functionality"""
    try:
        print("\nTesting cost function functionality...")
        
        from cost_function_visualization import CostFunctionVisualizer, generate_sample_data
        
        # Generate sample data
        X, y = generate_sample_data(n_samples=10, noise_level=1.0, random_state=42)
        print("✅ Sample data generated")
        
        # Create visualizer
        visualizer = CostFunctionVisualizer(X, y)
        print("✅ CostFunctionVisualizer created")
        
        # Test cost computation
        cost = visualizer.compute_cost(1.0, 0.0)
        print(f"✅ Cost computation: {cost:.4f}")
        
        # Test gradient computation
        dj_dw, dj_db = visualizer.compute_gradient(1.0, 0.0)
        print(f"✅ Gradient computation: dj_dw={dj_dw:.4f}, dj_db={dj_db:.4f}")
        
        print("🎉 Cost function functionality works!")
        return True
        
    except Exception as e:
        print(f"❌ Cost function error: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("STREAMLIT CLOUD DEPLOYMENT TEST")
    print("="*60)
    
    success = test_imports()
    if success:
        test_cost_function()
    
    print("\n" + "="*60)
    if success:
        print("✅ READY FOR DEPLOYMENT!")
    else:
        print("❌ DEPLOYMENT ISSUES DETECTED")
    print("="*60)
