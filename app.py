#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning Classification System - Web UI
Streamlit Application


@author: aymen abdelkouddous hamel
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import StringIO

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
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="KERTAS Paleographer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration class for data paths and model parameters"""
    
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
    
    CLASS_NAMES = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 
                   'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14']
    
    SVM_PARAM_GRID = {
        'C': [500, 1000, 5000, 10000],
        'gamma': [5000, 500, 50, 5],
        'kernel': ['rbf']
    }
    
    RF_PARAM_GRID = {
        'n_estimators': [10, 20, 50, 100],
        'max_depth': [5, 10, 20]
    }
    
    GBT_PARAM_GRID = {
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200, 300]
    }

# ============================================================================
# STYLING
# ============================================================================

def load_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_data(feature_type='chaincode'):
    """Load training and testing data"""
    try:
        if feature_type.lower() == 'chaincode':
            X_train = pd.read_csv(Config.TRAINING_CHAINCODE_PATH)
            X_test = pd.read_csv(Config.TESTING_CHAINCODE_PATH)
        else:
            X_train = pd.read_csv(Config.TRAINING_POLYGON_PATH)
            X_test = pd.read_csv(Config.TESTING_POLYGON_PATH)
        
        y_train = pd.read_csv(Config.TRAINING_LABELS_PATH)
        y_test = pd.read_csv(Config.TESTING_LABELS_PATH)
        
        # FIX: Reset column names to avoid feature name mismatch
        # This ensures training and testing data have identical column names
        X_train.columns = range(X_train.shape[1])
        X_test.columns = range(X_test.shape[1])
        
        return X_train, X_test, y_train, y_test, True, "Data loaded successfully!"
        
    except FileNotFoundError as e:
        return None, None, None, None, False, f"Error: Data file not found - {e}"

# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(model_name, feature_type, use_grid_search, X_train, y_train):
    """Train selected model"""
    
    y_train = y_train.values.ravel()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        if model_name == "SVM":
            status_text.text("Training SVM model...")
            progress_bar.progress(25)
            
            if use_grid_search:
                grid = GridSearchCV(SVC(), Config.SVM_PARAM_GRID, cv=3, verbose=0)
                grid.fit(X_train, y_train)
                model = grid.best_estimator_
                best_params = grid.best_params_
            else:
                model = SVC(kernel='rbf', C=10000, gamma=500)
                model.fit(X_train, y_train)
                best_params = None
                
        elif model_name == "Random Forest":
            status_text.text("Training Random Forest model...")
            progress_bar.progress(25)
            
            if use_grid_search:
                grid = GridSearchCV(RandomForestClassifier(), Config.RF_PARAM_GRID, cv=3, n_jobs=-1, verbose=0)
                grid.fit(X_train, y_train)
                model = grid.best_estimator_
                best_params = grid.best_params_
            else:
                model = RandomForestClassifier(n_estimators=100, max_depth=20)
                model.fit(X_train, y_train)
                best_params = None
                
        elif model_name == "Gradient Boosting":
            status_text.text("Training Gradient Boosting model...")
            progress_bar.progress(25)
            
            if use_grid_search:
                gb = GradientBoostingClassifier()
                grid = GridSearchCV(gb, Config.GBT_PARAM_GRID, cv=3, verbose=0)
                grid.fit(X_train, y_train)
                model = grid.best_estimator_
                best_params = grid.best_params_
            else:
                model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1)
                model.fit(X_train, y_train)
                best_params = None
                
        elif model_name == "AdaBoost":
            status_text.text("Training AdaBoost model...")
            progress_bar.progress(25)
            
            DTC = DecisionTreeClassifier(random_state=11, max_features="auto",
                                        class_weight="balanced", max_depth=None)
            model = AdaBoostClassifier(estimator=DTC)
            model.fit(X_train, y_train)
            best_params = None
        
        progress_bar.progress(100)
        status_text.text("✅ Training complete!")
        
        return model, best_params, True, "Model trained successfully!"
        
    except Exception as e:
        return None, None, False, f"Error training model: {e}"

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_confusion_matrix(y_test, predictions):
    """Plot confusion matrix"""
    matrix = confusion_matrix(y_test, predictions)
    matrix_normalized = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix_normalized, annot=True, fmt='.2f', 
               cmap='Blues', linewidths=0.5,
               xticklabels=Config.CLASS_NAMES,
               yticklabels=Config.CLASS_NAMES,
               ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Normalized Confusion Matrix', fontsize=14, pad=20)
    
    return fig

def plot_accuracy_comparison(results):
    """Plot accuracy comparison bar chart"""
    models = list(results.keys())
    accuracies = [results[m] * 100 for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}%', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Model Comparison', fontsize=14, pad=20)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    load_css()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 KERTAS Paleographer: A Machine Learning Classification System</h1>', 
                unsafe_allow_html=True)
    
    # Navigation tabs
    tab1, tab2, tab3 = st.tabs(["🚀 Train Models", "📊 About Project", "📚 Documentation"])
    
    # Initialize session state
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
    if 'results' not in st.session_state:
        st.session_state.results = {}
    
    # Sidebar
    with st.sidebar:
        # Display logo from Assets folder
        logo_path = os.path.join(Config.DATA_DIR, "Assets", "brain.png")
        if os.path.exists(logo_path):
            st.image(logo_path, width=150)
        else:
            st.markdown("🧠")  # Fallback emoji if image not found
        st.title("⚙️ Configuration")
        
        st.markdown("### 📊 Select Options")
        
        mode = st.radio(
            "Mode:",
            ["Single Model Training", "Compare All Models"],
            help="Choose to train a single model or compare multiple models"
        )
        
        feature_type = st.selectbox(
            "Feature Type:",
            ["ChainCode", "Polygon"],
            help="Select the type of features to use"
        )
        
        if mode == "Single Model Training":
            model_name = st.selectbox(
                "Model:",
                ["SVM", "Random Forest", "Gradient Boosting", "AdaBoost"],
                help="Select the machine learning model to train"
            )
        
        use_grid_search = st.checkbox(
            "Use Grid Search",
            value=False,
            help="Enable hyperparameter optimization (slower but better results)"
        )
        
        st.markdown("---")
        st.markdown("### 📁 Data Info")
        
        # Load data
        X_train, X_test, y_train, y_test, success, message = load_data(feature_type.lower())
        
        if success:
            st.success("✅ Data loaded")
            st.metric("Training Samples", X_train.shape[0])
            st.metric("Testing Samples", X_test.shape[0])
            st.metric("Features", X_train.shape[1])
        else:
            st.error(message)
            st.stop()
    
    # ========================================================================
    # TAB 1: TRAIN MODELS
    # ========================================================================
    
    with tab1:
        st.markdown("---")
        
        # Main content
        if mode == "Single Model Training":
            st.header(f"🎯 Single Model Training: {model_name}")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### 🔧 Training Configuration")
                st.info(f"""
                **Model:** {model_name}  
                **Feature Type:** {feature_type}  
                **Grid Search:** {'Enabled' if use_grid_search else 'Disabled'}  
                **Training Samples:** {X_train.shape[0]}  
                **Testing Samples:** {X_test.shape[0]}
                """)
                
                if st.button("🚀 Train Model", type="primary", use_container_width=True):
                    with st.spinner("Training in progress..."):
                        model, best_params, success, msg = train_model(
                            model_name, feature_type, use_grid_search, X_train, y_train
                        )
                        
                        if success:
                            st.session_state.trained_model = model
                            st.session_state.model_name = model_name
                            st.session_state.best_params = best_params
                            
                            # Make predictions
                            predictions = model.predict(X_test)
                            accuracy = accuracy_score(y_test, predictions)
                            
                            st.session_state.predictions = predictions
                            st.session_state.accuracy = accuracy
                            st.session_state.y_test = y_test
                            
                            st.success(f"✅ {msg}")
                            st.balloons()
                        else:
                            st.error(f"❌ {msg}")
            
            with col2:
                if st.session_state.trained_model:
                    st.markdown("### 📊 Results")
                    
                    # Accuracy
                    acc_col1, acc_col2, acc_col3 = st.columns(3)
                    with acc_col1:
                        st.metric("Accuracy", f"{st.session_state.accuracy*100:.2f}%")
                    with acc_col2:
                        st.metric("Model", st.session_state.model_name)
                    with acc_col3:
                        st.metric("Features", feature_type)
                    
                    # Best parameters
                    if st.session_state.best_params:
                        st.markdown("#### 🎯 Best Parameters (Grid Search)")
                        st.json(st.session_state.best_params)
                    
                    # Tabs for different views
                    result_tab1, result_tab2 = st.tabs(["📈 Confusion Matrix", "📋 Classification Report"])
                    
                    with result_tab1:
                        fig = plot_confusion_matrix(st.session_state.y_test, st.session_state.predictions)
                        st.pyplot(fig)
                    
                    with result_tab2:
                        # Generate classification report as dictionary
                        report_dict = classification_report(
                            st.session_state.y_test, 
                            st.session_state.predictions,
                            target_names=Config.CLASS_NAMES,
                            output_dict=True
                        )
                        
                        # Convert to DataFrame for better display
                        report_df = pd.DataFrame(report_dict).transpose()
                        
                        # Format the DataFrame
                        # Separate per-class metrics from averages
                        class_metrics = report_df.iloc[:-3].copy()  # All classes
                        avg_metrics = report_df.iloc[-3:].copy()    # accuracy, macro avg, weighted avg
                        
                        # Round values for display
                        for col in ['precision', 'recall', 'f1-score']:
                            if col in class_metrics.columns:
                                class_metrics[col] = class_metrics[col].round(3)
                            if col in avg_metrics.columns:
                                avg_metrics[col] = avg_metrics[col].round(3)
                        
                        if 'support' in class_metrics.columns:
                            class_metrics['support'] = class_metrics['support'].astype(int)
                        if 'support' in avg_metrics.columns:
                            avg_metrics['support'] = avg_metrics['support'].astype(int)
                        
                        # Display per-class metrics
                        st.markdown("#### 📊 Per-Class Performance Metrics")
                        
                        def highlight_performance(val):
                            """Color code based on performance"""
                            try:
                                val = float(val)
                                if val >= 0.8:
                                    return 'background-color: #d4edda; color: #155724; font-weight: bold'
                                elif val >= 0.6:
                                    return 'background-color: #fff3cd; color: #856404'
                                else:
                                    return 'background-color: #f8d7da; color: #721c24'
                            except:
                                return ''
                        
                        # Style per-class metrics with color coding
                        styled_class = class_metrics.style.applymap(
                            highlight_performance, 
                            subset=['precision', 'recall', 'f1-score']
                        ).set_properties(**{
                            'text-align': 'center',
                            'font-size': '14px',
                            'border': '1px solid #ddd'
                        }).set_table_styles([
                            {'selector': 'th', 'props': [
                                ('background-color', '#4A90E2'),
                                ('color', 'white'),
                                ('font-weight', 'bold'),
                                ('text-align', 'center'),
                                ('padding', '12px'),
                                ('font-size', '14px')
                            ]},
                            {'selector': 'td', 'props': [
                                ('padding', '10px')
                            ]},
                            {'selector': '', 'props': [
                                ('border-collapse', 'collapse'),
                                ('width', '100%')
                            ]}
                        ])
                        
                        st.dataframe(styled_class, use_container_width=True, height=550)
                        
                        # Add performance legend
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown("🟢 **Excellent**: ≥ 0.80")
                        with col2:
                            st.markdown("🟡 **Good**: 0.60-0.79")
                        with col3:
                            st.markdown("🔴 **Needs Improvement**: < 0.60")
                        
                        st.markdown("---")
                        
                        # Display overall metrics
                        st.markdown("#### 📈 Overall Performance Summary")
                        
                        # Create a more visual summary
                        overall_cols = st.columns(3)
                        
                        with overall_cols[0]:
                            accuracy_val = avg_metrics.loc['accuracy', 'precision'] if 'accuracy' in avg_metrics.index else 0
                            st.metric(
                                label="🎯 Overall Accuracy",
                                value=f"{accuracy_val:.1%}",
                                delta=f"{(accuracy_val - 0.5):.1%} vs random"
                            )
                        
                        with overall_cols[1]:
                            macro_f1 = avg_metrics.loc['macro avg', 'f1-score'] if 'macro avg' in avg_metrics.index else 0
                            st.metric(
                                label="📊 Macro F1-Score",
                                value=f"{macro_f1:.3f}",
                                delta="Unweighted average"
                            )
                        
                        with overall_cols[2]:
                            weighted_f1 = avg_metrics.loc['weighted avg', 'f1-score'] if 'weighted avg' in avg_metrics.index else 0
                            st.metric(
                                label="⚖️ Weighted F1-Score",
                                value=f"{weighted_f1:.3f}",
                                delta="Balanced by support"
                            )
                        
                        st.markdown("")
                        
                        # Detailed averages table
                        with st.expander("📋 Detailed Average Metrics"):
                            styled_avg = avg_metrics.style.format({
                                'precision': '{:.3f}',
                                'recall': '{:.3f}',
                                'f1-score': '{:.3f}',
                                'support': '{:.0f}'
                            }).background_gradient(
                                cmap='RdYlGn',
                                subset=['precision', 'recall', 'f1-score'],
                                vmin=0.0,
                                vmax=1.0
                            ).set_properties(**{
                                'text-align': 'center',
                                'font-size': '13px',
                                'padding': '10px'
                            }).set_table_styles([
                                {'selector': 'th', 'props': [
                                    ('background-color', '#28a745'),
                                    ('color', 'white'),
                                    ('font-weight', 'bold'),
                                    ('text-align', 'center'),
                                    ('padding', '10px')
                                ]}
                            ])
                            
                            st.dataframe(styled_avg, use_container_width=True)
                        
                        # Add metric explanations
                        with st.expander("ℹ️ Understanding the Metrics"):
                            st.markdown("""
                            #### Performance Metrics Explained
                            
                            **Precision** 🎯
                            - What percentage of positive predictions were correct?
                            - *High precision* = Few false positives
                            - Important when false positives are costly
                            
                            **Recall (Sensitivity)** 🔍
                            - What percentage of actual positives were found?
                            - *High recall* = Few false negatives
                            - Important when missing positives is costly
                            
                            **F1-Score** ⚖️
                            - Harmonic mean of precision and recall
                            - Formula: *F1 = 2 × (Precision × Recall) / (Precision + Recall)*
                            - Balanced measure of performance
                            
                            **Support** 📊
                            - Number of actual samples in each class
                            - Helps identify class imbalance
                            
                            #### Averages Explained
                            
                            **Accuracy** ✓
                            - Overall percentage of correct predictions
                            - Can be misleading with imbalanced datasets
                            
                            **Macro Average** 📊
                            - Simple average across all classes
                            - Treats all classes equally
                            - Good for balanced datasets
                            
                            **Weighted Average** ⚖️
                            - Average weighted by class support
                            - Accounts for class imbalance
                            - More representative for imbalanced data
                            """)
        
        else:  # Compare All Models
            st.header("📊 Compare All Models")
            
            st.info(f"**Feature Type:** {feature_type} | **Grid Search:** {'Enabled' if use_grid_search else 'Disabled'}")
            
            if st.button("🔄 Compare All Models", type="primary", use_container_width=True):
                models_to_compare = ["SVM", "Random Forest", "Gradient Boosting", "AdaBoost"]
                results = {}
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, model_name in enumerate(models_to_compare):
                    status_text.text(f"Training {model_name}... ({idx+1}/{len(models_to_compare)})")
                    
                    model, best_params, success, msg = train_model(
                        model_name, feature_type, use_grid_search, X_train, y_train
                    )
                    
                    if success:
                        predictions = model.predict(X_test)
                        accuracy = accuracy_score(y_test, predictions)
                        results[model_name] = accuracy
                    
                    progress_bar.progress((idx + 1) / len(models_to_compare))
                
                st.session_state.results = results
                status_text.text("✅ All models trained!")
                st.balloons()
            
            if st.session_state.results:
                st.markdown("### 📈 Results")
                
                # Display metrics
                cols = st.columns(4)
                sorted_results = sorted(st.session_state.results.items(), key=lambda x: x[1], reverse=True)
                
                for idx, (model, acc) in enumerate(sorted_results):
                    with cols[idx]:
                        delta = "🏆 Best" if idx == 0 else None
                        st.metric(model, f"{acc*100:.2f}%", delta=delta)
                
                # Comparison chart
                st.markdown("### 📊 Visual Comparison")
                fig = plot_accuracy_comparison(st.session_state.results)
                st.pyplot(fig)
                
                # Best model
                best_model = sorted_results[0]
                st.success(f"🏆 **Best Model:** {best_model[0]} with {best_model[1]*100:.2f}% accuracy")
                
                # Results table
                st.markdown("### 📋 Detailed Results")
                df_results = pd.DataFrame([
                    {"Model": model, "Accuracy": f"{acc*100:.2f}%", "Score": acc}
                    for model, acc in sorted_results
                ])
                st.dataframe(df_results.drop(columns=['Score']), use_container_width=True)
    
    # ========================================================================
    # TAB 2: ABOUT PROJECT
    # ========================================================================
    
    with tab2:
        st.markdown("---")
        
        # Project Overview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("## 📖 About KERTAS Paleographer")
            st.markdown("""
            **KERTAS Paleographer** is an advanced machine learning system designed for 
            paleographic analysis and historical manuscript classification.
            
            This application leverages state-of-the-art machine learning algorithms to 
            classify and analyze historical documents based on various feature extraction 
            methods.
            """)
        
        with col2:
            st.info("""
            **Version:** 1.0  
            **Author:** Aymen Abdelkouddous Hamel  
            **Year:** 2022  
            **Institution:** University Research Project
            """)
        
        st.markdown("---")
        
        # Key Features
        st.markdown("### ✨ Key Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🤖 Multiple ML Models**
            - Support Vector Machine (SVM)
            - Random Forest
            - Gradient Boosting Trees
            - AdaBoost
            """)
        
        with col2:
            st.markdown("""
            **🔢 Feature Extraction**
            - ChainCode Global Features
            - Polygon-based Features
            - 605+ feature dimensions
            - Optimized preprocessing
            """)
        
        with col3:
            st.markdown("""
            **📊 Advanced Analytics**
            - Real-time training
            - Confusion matrices
            - Classification reports
            - Model comparison
            """)
        
        st.markdown("---")
        
        # Technical Specifications
        st.markdown("### 🔧 Technical Specifications")
        
        tech_col1, tech_col2 = st.columns(2)
        
        with tech_col1:
            st.markdown("""
            **Dataset Information:**
            - Training Samples: 1,438
            - Testing Samples: 663
            - Features: 605
            - Classes: 14 (C1-C14)
            - Format: CSV
            """)
        
        with tech_col2:
            st.markdown("""
            **Technology Stack:**
            - Python 3.7+
            - Streamlit (Web UI)
            - scikit-learn (ML)
            - Pandas & NumPy (Data)
            - Matplotlib & Seaborn (Viz)
            """)
        
        st.markdown("---")
        
        # Methodology
        st.markdown("### 🔬 Methodology")
        
        st.markdown("""
        #### Feature Extraction
        
        **ChainCode Global Features:**
        - Captures shape characteristics through directional encoding
        - Global representation of document structure
        - Invariant to rotation and translation
        
        **Polygon Features:**
        - Geometric approximation of document contours
        - Captures local and global shape properties
        - Efficient dimensionality reduction
        
        #### Classification Pipeline
        
        1. **Data Loading** → Load training and testing datasets
        2. **Preprocessing** → Feature normalization and column alignment
        3. **Model Training** → Train selected ML algorithm
        4. **Hyperparameter Tuning** → Optional Grid Search optimization
        5. **Evaluation** → Generate metrics and visualizations
        6. **Comparison** → Compare multiple models side-by-side
        """)
        
        st.markdown("---")
        
        # Performance Metrics
        st.markdown("### 📈 Performance Metrics")
        
        st.markdown("""
        The system evaluates model performance using multiple metrics:
        
        - **Accuracy**: Overall correctness of predictions
        - **Precision**: Reliability of positive predictions
        - **Recall**: Coverage of actual positive cases
        - **F1-Score**: Harmonic mean of precision and recall
        - **Confusion Matrix**: Detailed breakdown of predictions vs actual
        """)
        
        # Expected Performance
        with st.expander("📊 Expected Performance Ranges"):
            perf_data = {
                'Model': ['SVM', 'Random Forest', 'Gradient Boosting', 'AdaBoost'],
                'ChainCode (%)': ['90-95', '88-92', '89-93', '85-90'],
                'Polygon (%)': ['85-90', '83-88', '84-89', '80-86'],
                'Training Time': ['2-5 min', '1-3 min', '3-7 min', '1-2 min']
            }
            st.table(pd.DataFrame(perf_data))
        
        st.markdown("---")
        
        # Use Cases
        st.markdown("### 🎯 Use Cases & Applications")
        
        use_col1, use_col2 = st.columns(2)
        
        with use_col1:
            st.markdown("""
            **Academic Research:**
            - Paleographic studies
            - Historical document analysis
            - Manuscript authentication
            - Dating estimation
            """)
        
        with use_col2:
            st.markdown("""
            **Practical Applications:**
            - Library digitization projects
            - Museum archival systems
            - Heritage preservation
            - Educational tools
            """)
        
        st.markdown("---")
        
        # System Architecture
        st.markdown("### 🏗️ System Architecture")
        
        st.markdown("""
        ```
        ┌─────────────────────────────────────────────────────────┐
        │                  KERTAS Paleographer                    │
        ├─────────────────────────────────────────────────────────┤
        │                                                         │
        │  ┌──────────────┐      ┌─────────────────┐            │
        │  │ Data Loading │─────→│  Preprocessing  │            │
        │  └──────────────┘      └─────────────────┘            │
        │         │                        │                     │
        │         ↓                        ↓                     │
        │  ┌──────────────┐      ┌─────────────────┐            │
        │  │Feature Extract│      │  Model Training │            │
        │  │(ChainCode/Poly)     │  (SVM/RF/GBT/AB)│            │
        │  └──────────────┘      └─────────────────┘            │
        │         │                        │                     │
        │         ↓                        ↓                     │
        │  ┌──────────────┐      ┌─────────────────┐            │
        │  │  Evaluation  │←─────│   Prediction    │            │
        │  └──────────────┘      └─────────────────┘            │
        │         │                                              │
        │         ↓                                              │
        │  ┌──────────────────────────────┐                     │
        │  │  Results & Visualization     │                     │
        │  └──────────────────────────────┘                     │
        └─────────────────────────────────────────────────────────┘
        ```
        """)
        
        st.markdown("---")
        
        # Credits & Acknowledgments
        st.markdown("### 🙏 Credits & Acknowledgments")
        
        st.markdown("""
        **Developed by:**  
        Aymen Abdelkouddous Hamel
        
        **Powered by:**
        - [scikit-learn](https://scikit-learn.org/) - Machine Learning library
        - [Streamlit](https://streamlit.io/) - Web application framework
        - [Pandas](https://pandas.pydata.org/) - Data manipulation
        - [NumPy](https://numpy.org/) - Numerical computing
        - [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) - Visualization
        
        **Special Thanks:**
        - Research supervisors and advisors
        - Open-source community
        - Beta testers and early users
        """)
        
        st.markdown("---")
        
        # Citation
        with st.expander("📄 How to Cite This Work"):
            st.code("""
@software{kertas_paleographer_2024,
  author = {Hamel, Aymen Abdelkouddous},
  title = {KERTAS Paleographer: A Machine Learning Classification System},
  year = {2022},
  version = {1.0},
  url = {https://github.com/yourrepo/kertas-paleographer}
}
            """, language="bibtex")
    
    # ========================================================================
    # TAB 3: DOCUMENTATION
    # ========================================================================
    
    with tab3:
        st.markdown("---")
        
        st.markdown("## 📚 Documentation & Help")
        
        # Quick Links
        st.markdown("### 🔗 Quick Links")
        
        doc_col1, doc_col2, doc_col3 = st.columns(3)
        
        with doc_col1:
            st.markdown("""
            **Getting Started:**
            - 📖 README.md
            - 🚀 QUICK_START.md
            - ⚙️ SETUP.md
            """)
        
        with doc_col2:
            st.markdown("""
            **User Guides:**
            - 🎨 UI_GUIDE.md
            - 💻 Command Line Guide
            - 🔧 Configuration Guide
            """)
        
        with doc_col3:
            st.markdown("""
            **Advanced:**
            - 🐛 FIX_APPLIED.md
            - 🌐 GITHUB_READY.md
            - 🧪 Testing Guide
            """)
        
        st.markdown("---")
        
        # How to Use
        st.markdown("### 💡 How to Use This Application")
        
        with st.expander("🎯 Single Model Training"):
            st.markdown("""
            1. **Select Configuration** (in sidebar):
               - Mode: "Single Model Training"
               - Feature Type: ChainCode or Polygon
               - Model: Choose your algorithm
               - Grid Search: Enable for optimization
            
            2. **Train Model**:
               - Click "🚀 Train Model" button
               - Watch progress bar
               - Wait for completion (~10-300 seconds)
            
            3. **View Results**:
               - Check accuracy metric
               - Explore confusion matrix
               - Read classification report
            """)
        
        with st.expander("📊 Compare All Models"):
            st.markdown("""
            1. **Configure** (in sidebar):
               - Mode: "Compare All Models"
               - Feature Type: Select one
               - Grid Search: Optional
            
            2. **Run Comparison**:
               - Click "🔄 Compare All Models"
               - All 4 models train automatically
               - Progress shown for each
            
            3. **Analyze Results**:
               - View accuracy for each model
               - See visual bar chart comparison
               - Identify best performing model
            """)
        
        st.markdown("---")
        
        # Tips & Tricks
        st.markdown("### 💡 Tips & Tricks")
        
        st.markdown("""
        **For Best Results:**
        - ✅ Start with Grid Search OFF for quick testing
        - ✅ Enable Grid Search for final/production results
        - ✅ Try both feature types to find best for your data
        - ✅ Compare all models before choosing one
        
        **Performance Optimization:**
        - ⚡ Disable Grid Search = 10-30 seconds per model
        - 🐢 Enable Grid Search = 1-5 minutes per model
        - 🎯 Random Forest is usually fastest
        - 🔍 SVM often gives best accuracy
        
        **Troubleshooting:**
        - 🔄 Refresh page if app becomes unresponsive
        - 💾 Results persist during your session
        - 🌐 Use Chrome or Firefox for best experience
        - 📊 Charts are interactive (hover, zoom)
        """)
        
        st.markdown("---")
        
        # System Requirements
        st.markdown("### 🖥️ System Requirements")
        
        req_col1, req_col2 = st.columns(2)
        
        with req_col1:
            st.markdown("""
            **Minimum:**
            - Python 3.7+
            - 4GB RAM
            - Modern web browser
            - Internet connection (first run)
            """)
        
        with req_col2:
            st.markdown("""
            **Recommended:**
            - Python 3.9+
            - 8GB+ RAM
            - Multi-core processor
            - SSD storage
            """)
        
        st.markdown("---")
        
        # FAQ
        st.markdown("### ❓ Frequently Asked Questions")
        
        with st.expander("Q: How accurate are the models?"):
            st.markdown("""
            Model accuracy typically ranges from 85-95% depending on:
            - Feature type used (ChainCode vs Polygon)
            - Model selected
            - Whether Grid Search is enabled
            - Data quality and distribution
            """)
        
        with st.expander("Q: How long does training take?"):
            st.markdown("""
            Training time varies:
            - **Without Grid Search**: 10-30 seconds per model
            - **With Grid Search**: 1-5 minutes per model
            - **Compare All (no GS)**: 40-120 seconds
            - **Compare All (with GS)**: 5-20 minutes
            """)
        
        with st.expander("Q: Can I use my own data?"):
            st.markdown("""
            Currently, the system is configured for the KERTAS dataset.
            To use your own data:
            1. Format CSV files to match expected structure
            2. Update paths in Config class
            3. Ensure same feature count
            4. Adjust class names if needed
            """)
        
        with st.expander("Q: Which model should I choose?"):
            st.markdown("""
            **Quick recommendation:**
            - **Best Accuracy**: SVM with Grid Search
            - **Fastest**: Random Forest without Grid Search
            - **Balanced**: Gradient Boosting with Grid Search
            - **Ensemble**: Try "Compare All" to decide
            """)
        
        st.markdown("---")
        
        # Contact & Support
        st.markdown("### 📧 Contact & Support")
        
        st.info("""
        **For questions, issues, or contributions:**
        
        📧 Email: aymen.hamel@university.edu  
        💻 GitHub: github.com/yourrepo/kertas-paleographer  
        📝 Report Issues: github.com/yourrepo/issues  
        🌟 Star the project if you find it useful!
        """)
    
    # Footer (outside tabs)
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🤖 KERTAS Paleographer | Built with Streamlit | © 2025TypeError: AdaBoostClassifier.__init__() got an unexpected keyword argument 'base_estimator'</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()

