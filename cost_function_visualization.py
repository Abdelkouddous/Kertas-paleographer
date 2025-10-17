#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cost Function Visualization for Machine Learning
Demonstrates 2D and 3D plots of cost function J(w,b) during training

@author: aymen abdelkouddous hamel
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import streamlit as st
import pandas as pd
from typing import Tuple, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CostFunctionVisualizer:
    """
    A comprehensive class for visualizing cost functions in machine learning
    Supports both 2D and 3D visualizations of parameters w, b and cost J(w,b)
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize the cost function visualizer
        
        Args:
            X: Feature matrix (m, n) where m is number of samples, n is number of features
            y: Target vector (m,)
        """
        self.X = X
        self.y = y
        self.m = X.shape[0]  # number of training examples
        self.n = X.shape[1]  # number of features
        
        # For simplicity, we'll work with single feature (n=1) for visualization
        if self.n > 1:
            print(f"Warning: Using only first feature for visualization. Original features: {self.n}")
            self.X = self.X[:, 0:1]  # Take only first feature
            self.n = 1
        
        # Initialize parameters
        self.w = 0.0  # weight
        self.b = 0.0  # bias
        self.cost_history = []
        self.w_history = []
        self.b_history = []
        
        # Normalize features for better convergence
        self.X_mean = np.mean(self.X)
        self.X_std = np.std(self.X)
        self.X_norm = (self.X - self.X_mean) / self.X_std
        
        # Normalize targets
        self.y_mean = np.mean(self.y)
        self.y_std = np.std(self.y)
        self.y_norm = (self.y - self.y_mean) / self.y_std
    
    def compute_cost(self, w: float, b: float, X: np.ndarray = None, y: np.ndarray = None) -> float:
        """
        Compute the cost function J(w,b) for linear regression
        
        Args:
            w: weight parameter
            b: bias parameter
            X: feature matrix (optional, uses normalized X if None)
            y: target vector (optional, uses normalized y if None)
            
        Returns:
            Cost value J(w,b)
        """
        if X is None:
            X = self.X_norm
        if y is None:
            y = self.y_norm
            
        m = X.shape[0]
        
        # Compute predictions
        f_wb = w * X.flatten() + b
        
        # Compute cost
        cost = (1 / (2 * m)) * np.sum((f_wb - y) ** 2)
        
        return cost
    
    def compute_gradient(self, w: float, b: float, X: np.ndarray = None, y: np.ndarray = None) -> Tuple[float, float]:
        """
        Compute gradients for w and b
        
        Args:
            w: weight parameter
            b: bias parameter
            X: feature matrix (optional)
            y: target vector (optional)
            
        Returns:
            Tuple of (dj_dw, dj_db)
        """
        if X is None:
            X = self.X_norm
        if y is None:
            y = self.y_norm
            
        m = X.shape[0]
        
        # Compute predictions
        f_wb = w * X.flatten() + b
        
        # Compute gradients
        dj_dw = (1 / m) * np.sum((f_wb - y) * X.flatten())
        dj_db = (1 / m) * np.sum(f_wb - y)
        
        return dj_dw, dj_db
    
    def gradient_descent(self, w_init: float = 0.0, b_init: float = 0.0, 
                        alpha: float = 0.01, num_iters: int = 1000) -> Tuple[List[float], List[float], List[float]]:
        """
        Perform gradient descent optimization
        
        Args:
            w_init: initial weight
            b_init: initial bias
            alpha: learning rate
            num_iters: number of iterations
            
        Returns:
            Tuple of (w_history, b_history, cost_history)
        """
        # Initialize parameters
        w = w_init
        b = b_init
        
        # Clear history
        self.w_history = []
        self.b_history = []
        self.cost_history = []
        
        for i in range(num_iters):
            # Compute cost and gradients
            cost = self.compute_cost(w, b)
            dj_dw, dj_db = self.compute_gradient(w, b)
            
            # Update parameters
            w = w - alpha * dj_dw
            b = b - alpha * dj_db
            
            # Store history
            self.w_history.append(w)
            self.b_history.append(b)
            self.cost_history.append(cost)
            
            # Print progress every 100 iterations
            if i % 100 == 0:
                print(f"Iteration {i:4d}: Cost {cost:.6f}, w = {w:.4f}, b = {b:.4f}")
        
        # Store final parameters
        self.w = w
        self.b = b
        
        print(f"\nFinal parameters: w = {w:.4f}, b = {b:.4f}")
        print(f"Final cost: {cost:.6f}")
        
        return self.w_history, self.b_history, self.cost_history
    
    def plot_2d_cost_contour(self, w_range: Tuple[float, float] = (-2, 2), 
                           b_range: Tuple[float, float] = (-2, 2), 
                           resolution: int = 50, 
                           show_trajectory: bool = True,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Create 2D contour plot showing cost function J(w,b)
        
        Args:
            w_range: range of w values to plot
            b_range: range of b values to plot
            resolution: resolution of the grid
            show_trajectory: whether to show gradient descent trajectory
            save_path: path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        # Create grid
        w_vals = np.linspace(w_range[0], w_range[1], resolution)
        b_vals = np.linspace(b_range[0], b_range[1], resolution)
        W, B = np.meshgrid(w_vals, b_vals)
        
        # Compute cost for each point
        J = np.zeros_like(W)
        for i in range(resolution):
            for j in range(resolution):
                J[i, j] = self.compute_cost(W[i, j], B[i, j])
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot contour
        contour = ax.contour(W, B, J, levels=20, alpha=0.6, colors='blue')
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.2f')
        
        # Fill contour
        contourf = ax.contourf(W, B, J, levels=50, alpha=0.7, cmap='viridis')
        plt.colorbar(contourf, ax=ax, label='Cost J(w,b)')
        
        # Plot trajectory if available
        if show_trajectory and len(self.w_history) > 0:
            ax.plot(self.w_history, self.b_history, 'ro-', linewidth=2, markersize=4, 
                   label='Gradient Descent Path', color='red')
            ax.plot(self.w_history[0], self.b_history[0], 'go', markersize=8, 
                   label='Start', color='green')
            ax.plot(self.w_history[-1], self.b_history[-1], 'r*', markersize=12, 
                   label='End', color='red')
        
        # Mark minimum
        min_idx = np.unravel_index(np.argmin(J), J.shape)
        ax.plot(W[min_idx], B[min_idx], 'y*', markersize=15, label='True Minimum', color='yellow')
        
        ax.set_xlabel('Weight (w)', fontsize=12)
        ax.set_ylabel('Bias (b)', fontsize=12)
        ax.set_title('Cost Function J(w,b) - 2D Contour Plot', fontsize=14, pad=20)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 2D contour plot saved to {save_path}")
        
        return fig
    
    def plot_3d_cost_surface(self, w_range: Tuple[float, float] = (-2, 2), 
                           b_range: Tuple[float, float] = (-2, 2), 
                           resolution: int = 30,
                           show_trajectory: bool = True,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Create 3D surface plot showing cost function J(w,b)
        
        Args:
            w_range: range of w values to plot
            b_range: range of b values to plot
            resolution: resolution of the grid
            show_trajectory: whether to show gradient descent trajectory
            save_path: path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        # Create grid
        w_vals = np.linspace(w_range[0], w_range[1], resolution)
        b_vals = np.linspace(b_range[0], b_range[1], resolution)
        W, B = np.meshgrid(w_vals, b_vals)
        
        # Compute cost for each point
        J = np.zeros_like(W)
        for i in range(resolution):
            for j in range(resolution):
                J[i, j] = self.compute_cost(W[i, j], B[i, j])
        
        # Create 3D plot
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot surface
        surface = ax.plot_surface(W, B, J, cmap='viridis', alpha=0.8, 
                                linewidth=0, antialiased=True)
        
        # Plot trajectory if available
        if show_trajectory and len(self.w_history) > 0:
            # Get cost values for trajectory
            trajectory_costs = [self.compute_cost(w, b) for w, b in zip(self.w_history, self.b_history)]
            
            ax.plot(self.w_history, self.b_history, trajectory_costs, 
                   'ro-', linewidth=3, markersize=6, label='Gradient Descent Path', color='red')
            ax.scatter(self.w_history[0], self.b_history[0], trajectory_costs[0], 
                      s=100, c='green', marker='o', label='Start')
            ax.scatter(self.w_history[-1], self.b_history[-1], trajectory_costs[-1], 
                      s=150, c='red', marker='*', label='End')
        
        # Mark minimum
        min_idx = np.unravel_index(np.argmin(J), J.shape)
        ax.scatter(W[min_idx], B[min_idx], J[min_idx], 
                  s=200, c='yellow', marker='*', label='True Minimum')
        
        ax.set_xlabel('Weight (w)', fontsize=12)
        ax.set_ylabel('Bias (b)', fontsize=12)
        ax.set_zlabel('Cost J(w,b)', fontsize=12)
        ax.set_title('Cost Function J(w,b) - 3D Surface Plot', fontsize=14, pad=20)
        ax.legend()
        
        # Add colorbar
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5, label='Cost J(w,b)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 3D surface plot saved to {save_path}")
        
        return fig
    
    def plot_cost_convergence(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot cost function convergence over iterations
        
        Args:
            save_path: path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        if not self.cost_history:
            raise ValueError("No cost history available. Run gradient_descent first.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot cost vs iterations
        ax1.plot(self.cost_history, 'b-', linewidth=2)
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Cost J(w,b)', fontsize=12)
        ax1.set_title('Cost Function Convergence', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # Log scale for better visualization
        
        # Plot parameter evolution
        ax2.plot(self.w_history, 'r-', linewidth=2, label='Weight (w)')
        ax2.plot(self.b_history, 'g-', linewidth=2, label='Bias (b)')
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Parameter Value', fontsize=12)
        ax2.set_title('Parameter Evolution', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Convergence plot saved to {save_path}")
        
        return fig
    
    def plot_data_and_fit(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot original data and fitted line
        
        Args:
            save_path: path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot original data
        ax.scatter(self.X, self.y, alpha=0.6, s=50, label='Training Data', color='blue')
        
        # Plot fitted line
        X_plot = np.linspace(self.X.min(), self.X.max(), 100).reshape(-1, 1)
        X_plot_norm = (X_plot - self.X_mean) / self.X_std
        y_pred_norm = self.w * X_plot_norm.flatten() + self.b
        
        # Denormalize predictions
        y_pred = y_pred_norm * self.y_std + self.y_mean
        
        ax.plot(X_plot, y_pred, 'r-', linewidth=2, label=f'Fitted Line (w={self.w:.3f}, b={self.b:.3f})')
        
        ax.set_xlabel('Feature X', fontsize=12)
        ax.set_ylabel('Target y', fontsize=12)
        ax.set_title('Linear Regression Fit', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Data and fit plot saved to {save_path}")
        
        return fig
    
    def create_interactive_3d_plot(self) -> go.Figure:
        """
        Create interactive 3D plot using Plotly
        
        Returns:
            plotly Figure object
        """
        # Create grid
        w_vals = np.linspace(-2, 2, 30)
        b_vals = np.linspace(-2, 2, 30)
        W, B = np.meshgrid(w_vals, b_vals)
        
        # Compute cost for each point
        J = np.zeros_like(W)
        for i in range(30):
            for j in range(30):
                J[i, j] = self.compute_cost(W[i, j], B[i, j])
        
        # Create 3D surface
        fig = go.Figure(data=[go.Surface(
            x=W,
            y=B,
            z=J,
            colorscale='Viridis',
            name='Cost Surface'
        )])
        
        # Add trajectory if available
        if len(self.w_history) > 0:
            trajectory_costs = [self.compute_cost(w, b) for w, b in zip(self.w_history, self.b_history)]
            
            fig.add_trace(go.Scatter3d(
                x=self.w_history,
                y=self.b_history,
                z=trajectory_costs,
                mode='markers+lines',
                marker=dict(size=4, color='red'),
                line=dict(color='red', width=4),
                name='Gradient Descent Path'
            ))
        
        fig.update_layout(
            title='Interactive 3D Cost Function J(w,b)',
            scene=dict(
                xaxis_title='Weight (w)',
                yaxis_title='Bias (b)',
                zaxis_title='Cost J(w,b)'
            ),
            width=800,
            height=600
        )
        
        return fig

def generate_sample_data(n_samples: int = 100, noise_level: float = 0.1, 
                        random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sample data for demonstration
    
    Args:
        n_samples: number of samples
        noise_level: level of noise to add
        random_state: random seed
        
    Returns:
        Tuple of (X, y) arrays
    """
    np.random.seed(random_state)
    
    # Generate X values
    X = np.random.uniform(-5, 5, n_samples).reshape(-1, 1)
    
    # Generate y values with some noise
    true_w = 2.5
    true_b = 1.0
    y = true_w * X.flatten() + true_b + np.random.normal(0, noise_level, n_samples)
    
    return X, y

def demonstrate_cost_function():
    """
    Demonstrate cost function visualization with sample data
    """
    print("="*70)
    print("COST FUNCTION VISUALIZATION DEMONSTRATION")
    print("="*70)
    
    # Generate sample data
    print("\n1. Generating sample data...")
    X, y = generate_sample_data(n_samples=50, noise_level=2.0)
    print(f"   Generated {X.shape[0]} samples with {X.shape[1]} feature(s)")
    
    # Create visualizer
    print("\n2. Creating cost function visualizer...")
    visualizer = CostFunctionVisualizer(X, y)
    
    # Run gradient descent
    print("\n3. Running gradient descent...")
    w_history, b_history, cost_history = visualizer.gradient_descent(
        w_init=0.0, b_init=0.0, alpha=0.1, num_iters=1000
    )
    
    # Create visualizations
    print("\n4. Creating visualizations...")
    
    # 2D contour plot
    print("   - Creating 2D contour plot...")
    fig_2d = visualizer.plot_2d_cost_contour(
        w_range=(-3, 3), b_range=(-3, 3), show_trajectory=True
    )
    plt.show()
    
    # 3D surface plot
    print("   - Creating 3D surface plot...")
    fig_3d = visualizer.plot_3d_cost_surface(
        w_range=(-3, 3), b_range=(-3, 3), show_trajectory=True
    )
    plt.show()
    
    # Convergence plot
    print("   - Creating convergence plot...")
    fig_conv = visualizer.plot_cost_convergence()
    plt.show()
    
    # Data and fit plot
    print("   - Creating data and fit plot...")
    fig_fit = visualizer.plot_data_and_fit()
    plt.show()
    
    print("\n✓ All visualizations completed!")
    print("="*70)

def streamlit_cost_function_demo():
    """
    Streamlit interface for cost function visualization
    """
    st.set_page_config(
        page_title="Cost Function Visualization",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Cost Function Visualization")
    st.markdown("Interactive visualization of cost function J(w,b) for linear regression")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Controls")
    
    # Data generation parameters
    st.sidebar.subheader("📊 Data Parameters")
    n_samples = st.sidebar.slider("Number of samples", 20, 200, 50)
    noise_level = st.sidebar.slider("Noise level", 0.1, 5.0, 2.0, 0.1)
    random_state = st.sidebar.number_input("Random seed", 0, 100, 42)
    
    # Training parameters
    st.sidebar.subheader("🚀 Training Parameters")
    learning_rate = st.sidebar.slider("Learning rate (α)", 0.001, 1.0, 0.1, 0.001)
    num_iterations = st.sidebar.slider("Number of iterations", 100, 5000, 1000, 100)
    
    # Visualization parameters
    st.sidebar.subheader("📈 Visualization Parameters")
    w_range = st.sidebar.slider("Weight range", -5.0, 5.0, (-3.0, 3.0))
    b_range = st.sidebar.slider("Bias range", -5.0, 5.0, (-3.0, 3.0))
    show_trajectory = st.sidebar.checkbox("Show gradient descent trajectory", True)
    
    # Generate data
    if st.sidebar.button("🔄 Generate New Data"):
        st.rerun()
    
    # Generate sample data
    X, y = generate_sample_data(n_samples=n_samples, noise_level=noise_level, random_state=random_state)
    
    # Create visualizer
    visualizer = CostFunctionVisualizer(X, y)
    
    # Run gradient descent
    with st.spinner("Running gradient descent..."):
        w_history, b_history, cost_history = visualizer.gradient_descent(
            w_init=0.0, b_init=0.0, alpha=learning_rate, num_iters=num_iterations
        )
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Final Weight (w)", f"{visualizer.w:.4f}")
    with col2:
        st.metric("Final Bias (b)", f"{visualizer.b:.4f}")
    with col3:
        st.metric("Final Cost", f"{cost_history[-1]:.6f}")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data & Fit", "🎯 2D Contour", "🌐 3D Surface", 
        "📈 Convergence", "🔄 Interactive 3D"
    ])
    
    with tab1:
        st.subheader("Data and Linear Fit")
        fig_fit = visualizer.plot_data_and_fit()
        st.pyplot(fig_fit)
    
    with tab2:
        st.subheader("2D Cost Function Contour")
        fig_2d = visualizer.plot_2d_cost_contour(
            w_range=w_range, b_range=b_range, show_trajectory=show_trajectory
        )
        st.pyplot(fig_2d)
    
    with tab3:
        st.subheader("3D Cost Function Surface")
        fig_3d = visualizer.plot_3d_cost_surface(
            w_range=w_range, b_range=b_range, show_trajectory=show_trajectory
        )
        st.pyplot(fig_3d)
    
    with tab4:
        st.subheader("Cost Function Convergence")
        fig_conv = visualizer.plot_cost_convergence()
        st.pyplot(fig_conv)
    
    with tab5:
        st.subheader("Interactive 3D Visualization")
        fig_interactive = visualizer.create_interactive_3d_plot()
        st.plotly_chart(fig_interactive, use_container_width=True)
    
    # Additional information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown("""
    This visualization demonstrates:
    - **Cost Function**: J(w,b) = (1/2m) Σ(f_wb - y)²
    - **Gradient Descent**: Optimization algorithm
    - **Parameters**: w (weight) and b (bias)
    - **Convergence**: How cost decreases over iterations
    """)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "streamlit":
        # Run Streamlit app
        import subprocess
        subprocess.run(["streamlit", "run", __file__])
    else:
        # Run demonstration
        demonstrate_cost_function()

