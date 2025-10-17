#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cost Function Visualization Demo
Simple demonstration of how to use the cost function visualization

@author: aymen abdelkouddous hamel
"""

import numpy as np
import matplotlib.pyplot as plt
from cost_function_visualization import CostFunctionVisualizer, generate_sample_data

def main():
    """
    Demonstrate cost function visualization with sample data
    """
    print("="*70)
    print("COST FUNCTION VISUALIZATION DEMONSTRATION")
    print("="*70)
    
    # Generate sample data
    print("\n1. Generating sample data...")
    X, y = generate_sample_data(n_samples=50, noise_level=2.0, random_state=42)
    print(f"   Generated {X.shape[0]} samples with {X.shape[1]} feature(s)")
    print(f"   X range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"   y range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Create visualizer
    print("\n2. Creating cost function visualizer...")
    visualizer = CostFunctionVisualizer(X, y)
    
    # Run gradient descent
    print("\n3. Running gradient descent...")
    print("   Parameters: α=0.1, iterations=1000")
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
    plt.savefig('cost_function_2d.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3D surface plot
    print("   - Creating 3D surface plot...")
    fig_3d = visualizer.plot_3d_cost_surface(
        w_range=(-3, 3), b_range=(-3, 3), show_trajectory=True
    )
    plt.savefig('cost_function_3d.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Convergence plot
    print("   - Creating convergence plot...")
    fig_conv = visualizer.plot_cost_convergence()
    plt.savefig('cost_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Data and fit plot
    print("   - Creating data and fit plot...")
    fig_fit = visualizer.plot_data_and_fit()
    plt.savefig('data_and_fit.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✓ All visualizations completed!")
    print("✓ Plots saved as PNG files in current directory")
    print("="*70)
    
    # Print summary
    print("\nSUMMARY:")
    print(f"Final parameters: w = {visualizer.w:.4f}, b = {visualizer.b:.4f}")
    print(f"Final cost: {cost_history[-1]:.6f}")
    print(f"Cost improvement: {((cost_history[0] - cost_history[-1]) / cost_history[0] * 100):.2f}%")
    print(f"Total iterations: {len(cost_history)}")

if __name__ == "__main__":
    main()

